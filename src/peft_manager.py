# src/peft_manager.py

from peft import LoraConfig, get_peft_model

class PEFTManager:
    @staticmethod
    def apply_peft(model, use_peft: str, scale_peft: float = 1.0):
        if use_peft is not None:
            if use_peft == 'lora':
                config = PEFTManager.get_peft_config(use_peft)
                lora_model = get_peft_model(model, config)
                if scale_peft is not None and scale_peft != 1.0 and scale_peft > 0:
                    for name, param in lora_model.named_parameters():
                        if "lora" in name:
                            param.data *= scale_peft
                return lora_model
            elif use_peft == 'prompt-tuning':
                # TO DO: Implement when available
                return model
            elif use_peft == 'adapter-tuning':
                # TO DO: Implement when available
                return model
            else:
                raise ValueError(f"PEFT method '{use_peft}' is not supported.")
        return model

    @staticmethod
    def get_peft_config(peft_type: str):
        if peft_type == 'lora':
            target_modules = ['c_attn', 'c_proj']
            task_type = 'CAUSAL_LM'
            return LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=target_modules,
                task_type=task_type
            )
        else:
            raise ValueError(f"PEFT method '{peft_type}' is not supported.")