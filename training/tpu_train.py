import os
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding
import optax
from transformers import AutoTokenizer, FlaxLlamaForCausalLM, LlamaConfig
from datasets import load_dataset
from tqdm.auto import tqdm
import wandb

# =============================================================================
# AfriLION TPU Training Loop (JAX/Flax)
# Optimized for Cloud TPU v4-32 / v5e-64 / v6e-64
# =============================================================================

def train():
    # 1. Configuration
    PROJECT_ID = "afrilion-49361e"
    MODEL_NAME = "LocaleNLP/afrilion-base"
    DATASET_NAME = "LocaleNLP/AfriCorpus-v1"
    BATCH_SIZE = 128  # Total batch size across all TPU cores
    SEQ_LENGTH = 2048
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    TOTAL_STEPS = 50000
    WARMUP_STEPS = 2000
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 1000
    OUTPUT_DIR = "gs://afrilion-tpu-afrilion-49361e/checkpoints"

    # 2. Setup TPU Mesh
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=('data',))
    print(f"Initialized TPU Mesh: {mesh}")
    print(f"Device count: {jax.device_count()}")

    # 3. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = LlamaConfig.from_pretrained(MODEL_NAME)
    model = FlaxLlamaForCausalLM(config, seed=42)
    
    # 4. Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=TOTAL_STEPS,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
    )
    opt_state = optimizer.init(model.params)

    # 5. Data Pipeline (TPU-optimized)
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        tokens = tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=SEQ_LENGTH, 
            return_tensors="np"
        )
        return tokens["input_ids"]

    # 6. Training Step (JIT-compiled)
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = model(batch, params=params).logits
            # Standard Causal LM loss
            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]
            loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return loss, state._replace(params=new_params, opt_state=new_opt_state)

    # 7. Initialize State
    class TrainState:
        def __init__(self, params, opt_state):
            self.params = params
            self.opt_state = opt_state
        def _replace(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            return self

    state = TrainState(model.params, opt_state)

    # 8. Training Loop
    wandb.init(project="afrilion", name="tpu-pretrain-v1")
    
    it = iter(dataset)
    pbar = tqdm(range(TOTAL_STEPS), desc="Training AfriLION")
    
    for step in pbar:
        # Fetch batch
        batch_list = []
        for _ in range(BATCH_SIZE):
            try:
                batch_list.append(next(it))
            except StopIteration:
                it = iter(dataset)
                batch_list.append(next(it))
        
        batch = collate_fn(batch_list)
        
        # Sharding
        sharding = NamedSharding(mesh, P('data', None))
        batch = jax.device_put(batch, sharding)
        
        # Step
        loss, state = train_step(state, batch)
        
        # Logging
        if step % LOG_INTERVAL == 0:
            loss_val = jax.device_get(loss)
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            wandb.log({"loss": loss_val, "step": step})

        # Checkpoint
        if step > 0 and step % SAVE_INTERVAL == 0:
            save_path = f"{OUTPUT_DIR}/step_{step}"
            print(f"Saving checkpoint to {save_path}...")
            # Note: In real setup, use Orbax or Flax Checkpointer for GCS
            # model.save_pretrained(save_path, params=state.params)

    wandb.finish()

if __name__ == "__main__":
    train()
