from diffusers.pipelines.sana.pipeline_sana import SanaPipeline
from sampling import sample_prompts

def gen_reference_dataset(pipe: SanaPipeline, prompts: list[dict], ref_dir: str, num_seeds: int):
    sample_prompts(
        pipe, [p["prompt"] for p in prompts], ref_dir,
        seed=0, num_seeds=num_seeds,
    )
