# LoRA Adapter (temporary MMI bridge)

In the current experimental setup, the cloud model is accessed via a provider
and its internal weights cannot be modified directly.

A LoRA adapter may serve as a temporary bridge: a small local parameter set
that carries the plastic autobiographical memory signal into the model's
representation space during inference.

## Role

- The LoRA adapter is NOT the final Matryoshka memory.
- It is a temporary experimental mechanism to let MMI inject memory context
  into a cloud-hosted model that cannot be internally modified.
- In the future built-in form, this bridge would not be needed.

## Files

LoRA weight files are git-ignored.

Expected file types: .safetensors, .bin
