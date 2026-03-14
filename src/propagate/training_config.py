from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Stores training hyperparameters.
    
    Attributes
    ----------
    total_steps : int
        The total number of steps to train for.
    population_size : int
        The number of genomes evaluated per step during optimization.
    mirror : bool
        Whether or not to mirror genomes. Doubles the population if true.
    rank_norm_rewards : bool
        Whether to normalize the rewards by rank. For example, the worse genome becomes -0.5, the next genome is -0.4, and the best is 0.5.
    learning_rate : float
        The learning rate of the optimizer.
    perturb_scale : float
        The scale of the perturbation (sigma).
    warmup_steps : int
        The number of warmup steps.
    scheduler : str
        The learning rate scheduler to use.
    centered_eval : bool, optional
        Whether or not to do one extra eval on the unperturbed gradient. Enables centered eval and dyanmic perturbation scales. Defaults to True.
    pass_true_mean : bool, optional
        Whether or not to pass the centered eval mean to the optimizer as the true mean for gradient calculation when norm_by_mean is set to true. Defaults to False.
    dynamic_perturbation_target : float, optional
        The amount we want our perturbed rewards to drift from the center. Defaults to 0.1.
    dynamic_perturbation_smoothing_factor : float, optional 
        Whether or not to dynamically adjust the perturbation scale using the centered eval and a slightly modified version of the PSR rule that works with mirroring. Defaults to 0, which disables it, recommend <0.5.
    """
    total_steps: int
    population_size: int
    
    mirror: bool
    rank_norm_rewards: bool
    
    learning_rate: float
    perturb_scale: float
    
    lr_scheduler: str = "none"
    warmup_steps: int = 0
    
    centered_eval: bool = True
    pass_true_mean: bool = False 
    dynamic_perturb_target: float = 0.25
    dynamic_perturb_smoothing_factor: float = 0
    
    def __post_init__(self):
        if self.total_steps <= 0:
            raise ValueError(f"Total steps must be positive, got {self.total_steps}")
        if self.population_size <= 1:
            raise ValueError(f"Population must be greater than 1, got {self.population_size}")
        if self.pass_true_mean and not self.centered_eval:
            raise ValueError("Cannot pass the true mean without centered eval!")
        if self.pass_true_mean and self.mirror:
            raise ValueError("Mirrored training cannot use a centered eval mean!")
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.perturb_scale <= 0:
            raise ValueError(f"Perturb scale must be positive, got {self.perturb_scale}")
        if self.warmup_steps < 0:
            raise ValueError(f"Warmup steps must be positive, got {self.warmup_steps}")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(f"Warmup steps cannot be equal or greater than total steps!")
        if (self.lr_scheduler or "none").lower() not in ["none", "constant", "linear", "cosine", "exponential"]:
            raise ValueError(f"Unknown scheduler '{str(self.lr_scheduler)}'. Avaliable schedulers: none/constant, linear, cosine, exponential.")
        if self.dynamic_perturb_target <= 0:
            raise ValueError(f"Target perturb must be positive, got {self.dynamic_perturb_target}")
        if self.dynamic_perturb_smoothing_factor < 0:
            raise ValueError(f"Perturb smoothing factor must be positive, got {self.dynamic_perturb_smoothing_factor}")