import unittest
from types import SimpleNamespace

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4DsparkWeightMapping(CustomTestCase):
    def remap(self, name: str):
        model = SimpleNamespace(num_stages=3, confidence_head=None)
        return DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(model, name)

    def test_expert_scale_bias_is_preserved(self):
        expected = {
            "mtp.2.ffn.experts.255.w1.scale_bias": (
                "stages.2.mlp.experts.255.gate_proj.scale_bias"
            ),
            "mtp.2.ffn.experts.255.w2.scale_bias": (
                "stages.2.mlp.experts.255.down_proj.scale_bias"
            ),
            "mtp.2.ffn.experts.255.w3.scale_bias": (
                "stages.2.mlp.experts.255.up_proj.scale_bias"
            ),
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(self.remap(source), target)

    def test_standalone_scale_is_remapped(self):
        self.assertEqual(
            self.remap("mtp.0.attn.wq_a.scale"),
            "stages.0.self_attn.wq_a.weight_scale_inv",
        )
        self.assertEqual(
            self.remap("mtp.0.self_attn.wq_a.scale"),
            "stages.0.self_attn.wq_a.weight_scale_inv",
        )

    def test_mtp_local_vocab_modules_are_loaded(self):
        self.assertEqual(
            self.remap("mtp.0.embed.weight"),
            "embed_tokens.weight",
        )
        self.assertEqual(
            self.remap("mtp.0.embed_tokens.weight"),
            "embed_tokens.weight",
        )
        self.assertEqual(
            self.remap("mtp.2.head.weight"),
            "lm_head.weight",
        )
        self.assertEqual(
            self.remap("mtp.2.lm_head.weight"),
            "lm_head.weight",
        )

    def test_non_owner_vocab_copies_and_target_vocab_are_ignored(self):
        self.assertIsNone(self.remap("mtp.1.embed.weight"))
        self.assertIsNone(self.remap("mtp.0.head.weight"))
        self.assertIsNone(self.remap("embed.weight"))
        self.assertIsNone(self.remap("head.weight"))

    def test_modelslim_dspark_quant_description_aliases(self):
        config = ModelSlimConfig(
            {
                # Triggers the generic DeepSeek-V4 name preprocessing that
                # canonicalizes attn -> self_attn before DSpark aliases run.
                "hc_head_fn": "FLOAT",
                "mtp.0.attn.wq_a.weight": "W8A8_DYNAMIC",
                "mtp.1.ffn.experts.0.w1.weight": "W4A8_DYNAMIC",
                "mtp.1.ffn.experts.0.w2.weight": "W4A8_DYNAMIC",
                "mtp.1.ffn.experts.0.w3.weight": "W4A8_DYNAMIC",
                "mtp.2.markov_head.w1.weight": "FLOAT",
            }
        )
        expected = {
            "stages.0.self_attn.wq_a.weight": "W8A8_DYNAMIC",
            "stages.1.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
            "stages.1.mlp.experts.0.down_proj.weight": "W4A8_DYNAMIC",
            "stages.1.mlp.experts.0.up_proj.weight": "W4A8_DYNAMIC",
            "markov_head.w1.weight": "FLOAT",
        }
        for name, scheme in expected.items():
            with self.subTest(name=name):
                self.assertEqual(config.quant_description.get(name), scheme)

        self.assertFalse(
            any(
                "self_self_attn" in name
                for name in config.quant_description
                if isinstance(name, str)
            )
        )


if __name__ == "__main__":
    unittest.main()
