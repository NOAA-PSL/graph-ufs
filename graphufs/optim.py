import optax

def clipped_cosine_adamw(
    n_linear,
    n_total,
    init_value=0.0,
    peak_value=1e-3,
    end_value=0.0,
    clip_grad_global_norm=32.,
    b1=0.9,
    b2=0.95,
    weight_decay=0.1,
):
    """Cosine decay schedule with linear warmup, using AdamW and global norm gradient clipping

    Args:
        n_linear, n_total (int): number of linear warmup and total optimization steps to take
        init_value, peak_value, end_value (float): Starting, peak, and final values of learning rate
        clip_grad_global_norm (float): threshold to clip gradient by it's global norm
        b1, b3, weight_decay (float): AdamW parameters
    """

    # define learning rate schedules
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=n_linear,
        decay_steps=n_total,
        end_value=end_value,
    )

    # Adam optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_grad_global_norm),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_schedule,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
        ),
    )
    return optimizer

