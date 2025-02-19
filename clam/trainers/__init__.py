from clam.trainers.action_decoder_trainer import ActionDecoderTrainer
from clam.trainers.bc_trainer import BCTrainer
from clam.trainers.clam_policy_trainer import CLAMPolicyTrainer
from clam.trainers.clam_trainer import CLAMTrainer
from clam.trainers.diffusion_clam_trainer import DiffusionCLAMTrainer
from clam.trainers.dynamo_trainer import DynaMoTrainer
from clam.trainers.lapa_trainer import LAPATrainer
from clam.trainers.vpt_bc_trainer import VPTBCTrainer
from clam.trainers.vpt_trainer import VPTTrainer

trainer_to_cls = {
    "clam": CLAMTrainer,
    "transformer_clam": CLAMTrainer,
    "diffusion_clam": DiffusionCLAMTrainer,
    "st_vivit_clam": CLAMTrainer,
    "clam_policy": CLAMPolicyTrainer,
    "bc": BCTrainer,
    "action_decoder": ActionDecoderTrainer,
    "lap": LAPATrainer,
    "lapa": LAPATrainer,
    "dynamo": DynaMoTrainer,
    "vpt": VPTTrainer,
    "vpt_bc": VPTBCTrainer,
}
