# src/peft_manager.py
"""
Module for applying Parameter-Efficient Fine-Tuning (PEFT) methods
to models.
"""
from peft import LoraConfig, get_peft_model


class PEFTManager:
    """
    Class providing static methods to apply PEFT configurations to models.
    """

    @staticmethod
    def apply_peft(model, use_peft: str) -> object:
        """
        Applies the specified PEFT method to the given model.

        Args:
            model: The model to which PEFT will be applied.
            use_peft (str): The PEFT method to use (e.g., 'lora').

        Returns:
            Model with PEFT applied.

        Raises:
            ValueError: If the specified PEFT method is not supported.
        """
        if use_peft == "lora":
            config = PEFTManager.get_peft_config(use_peft)
            lora_model = get_peft_model(model, config)
            return lora_model
        raise ValueError(f"PEFT method '{use_peft}' is not supported.")

    @staticmethod
    def get_peft_config(peft_type: str) -> LoraConfig:
        """
        Retrieves the PEFT configuration for the specified method.

        Args:
            peft_type (str): The PEFT method type.

        Returns:
            LoraConfig: The configuration object for LoRA.

        Raises:
            ValueError: If the specified PEFT method is not supported.
        """
        if peft_type == "lora":
            # target_modules = ["attn.c_attn", "attn.c_proj"]
            # target_modules = ["c_attn", "c_proj", "c_fc"]
            target_modules = "all-linear"     # options: "all-linear", "all-attention" or ["attn.c_attn", "attn.c_proj"]
            task_type = "CAUSAL_LM"
            return LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=target_modules,
                task_type=task_type,
                fan_in_fan_out=True,
                init_lora_weights=True    # options: True, False, 'gaussian', 'olora', 'pissa', 'loftq'
            )
        raise ValueError(f"PEFT method '{peft_type}' is not supported.")
