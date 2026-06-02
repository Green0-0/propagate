from propagate.optimizers import chain, chain_adam, chain_adam_seeded, chain_log, chain_misc
from propagate.optimizers.psamplers import Gaussian_PSampler, Bernoulli_PSampler

# PERTURBATIONS
STANDARD_GAUSSIAN_PERTURB_BF16 = [
    chain.Init_Perturbation(Gaussian_PSampler(fp32_accumulate=False)),
    chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

STANDARD_BERNOULLI_PERTURB_BF16 = [
    chain.Init_Perturbation(Bernoulli_PSampler(fp32_accumulate=False)),
    chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

STANDARD_GAUSSIAN_PERTURB = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

STANDARD_BERNOULLI_PERTURB = [
    chain.Init_Perturbation(Bernoulli_PSampler()),
    chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

NESTEROV_MOMENTUM_SEEDED_PERTURB = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain_adam_seeded.OC_Apply_Momentum_Seeded(Gaussian_PSampler()),
    chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

NESTEROV_RMSPROP_BLOCKWISE_SEEDED_PERTURB = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True, div_by_rmsprop_block=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

NESTEROV_ADAM_BLOCKWISE_SEEDED_PERTURB = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain_adam_seeded.OC_Apply_Momentum_Seeded(Gaussian_PSampler()),
    chain.Scale_Perturbation(mul_by_std=True, mul_by_lr_scalar=True, div_by_rmsprop_block=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

# UPDATES
STANDARD_GAUSSIAN_UPDATE = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

STANDARD_BERNOULLI_UPDATE = [
    chain.Init_Perturbation(Bernoulli_PSampler()),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

SIGNSGD_UPDATE = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain.Sign_Perturb_Buffer(),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

MOMENTUM_SEEDED_UPDATE = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain.Zero_Perturb_Buffer(),
    chain_adam_seeded.OC_Update_Seed_History(),
    chain_adam_seeded.OC_Apply_Momentum_Seeded(Gaussian_PSampler()),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

LION_SEEDED_UPDATE = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain.Zero_Perturb_Buffer(),
    chain_adam_seeded.OC_Update_Seed_History(),
    chain_adam_seeded.OC_Apply_Momentum_Seeded(Gaussian_PSampler()),
    chain.Sign_Perturb_Buffer(),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

ADAM_BLOCKWISE_SEEDED_UPDATE = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain_adam.OC_Compute_RMSProp_Blockwise(),
    chain.Zero_Perturb_Buffer(),
    chain_adam_seeded.OC_Update_Seed_History(),
    chain_adam_seeded.OC_Apply_Momentum_Seeded(Gaussian_PSampler()),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_rmsprop_block=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

MUON_SEEDED_UPDATE = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain.Zero_Perturb_Buffer(),
    chain_adam_seeded.OC_Update_Seed_History(),
    chain_adam_seeded.OC_Apply_Momentum_Seeded(Gaussian_PSampler()),
    chain_misc.OC_Muon_Whiten_Perturb_Buffer(),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

ADAMUON_BLOCKWISE_SEEDED_UPDATE = [
    chain.Init_Perturbation(Gaussian_PSampler()),
    chain_adam.OC_Compute_RMSProp_Blockwise(),
    chain.Zero_Perturb_Buffer(),
    chain_adam_seeded.OC_Update_Seed_History(),
    chain_adam_seeded.OC_Apply_Momentum_Seeded(Gaussian_PSampler()),
    chain_misc.OC_Muon_Whiten_Perturb_Buffer(),
    chain.Scale_Perturbation(mul_by_lr=True, mul_by_std=True, mul_by_lr_scalar=True, div_by_rmsprop_block=True, div_by_pop=True),
    chain.Add_Perturb_Buffer(),
    chain.Delete_Perturb_Buffer()
]

