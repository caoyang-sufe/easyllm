# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

from src.modules.qwen2 import (
	SkipLayerQwen2Model, SkipLayerQwen2ForCausalLM,
	ParallelQwen2Model, ParallelQwen2ForCausalLM,
)
from src.modules.qwen3 import (
	SkipLayerQwen3Model, SkipLayerQwen3ForCausalLM,
	ParallelQwen3Model, ParallelQwen3ForCausalLM,
)
from src.modules.llama import (
	SkipLayerLlamaModel, SkipLayerLlamaForCausalLM,
	ParallelLlamaModel, ParallelLlamaForCausalLM,
)

from src.modules.deepseek import (
	SkipLayerDeepseekModel, SkipLayerDeepseekForCausalLM,
	ParallelDeepseekModel, ParallelDeepseekForCausalLM,
)

from src.modules.deepseek_v2 import (
	SkipLayerDeepseekV2Model, SkipLayerDeepseekV2ForCausalLM,
	ParallelDeepseekV2Model, ParallelDeepseekV2ForCausalLM,
)

from src.modules.deepseek_v3 import (
	SkipLayerDeepseekV3Model, SkipLayerDeepseekV3ForCausalLM,
	ParallelDeepseekV3Model, ParallelDeepseekV3ForCausalLM,
)

