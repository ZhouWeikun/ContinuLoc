import torch
import torch.nn as nn
import torch.nn.functional as F
from continuloc.models.residual_mlp import ResnetBlockFC


class SerialModulatorShallow(nn.Module):
    """Shallow serial modulator conditioned on external features.

    Modulate inputs layer by layer using conditions such as plane or voxel
    features, for conditional generation, implicit representations, or occupancy.

    Args:
        input_dim: Input feature dimension, e.g. 3 for 3D coordinates.
        condition_dim: External conditioning feature dimension.
        hidden_dim: Hidden-layer dimension.
        num_blocks: Number of ResNet blocks.
        output_dim: Output dimension, default 1 for SDF/occupancy.
        condition_operator: Condition fusion, 'add' or 'mul'.
        output_activation_type: relu, leaky_relu, gelu, silu, or none before output.
        encoder_init_type: Input-layer initialization: default, kaiming_normal,
            xavier_uniform, zero, or small_random.
        output_init_type: Output-layer initialization with the same choices.
        condition_init_type: Condition-projection initialization with the same choices.
        block_norm_type: ResNet block normalization: none or layernorm.
        block_activation_type: Block activation: relu, leaky_relu, gelu, silu, or none.
        block_init_type: Block initialization: default, resnet_zero_init,
            kaiming_normal, or xavier_uniform."""

    def __init__(self,
                 input_dim: int = 3,
                 condition_dim: int = 128,
                 hidden_dim: int = 256,
                 num_blocks: int = 5,
                 output_dim: int = 1,
                 condition_operator: str = 'add',
                 output_activation_type: str = 'relu',
                 encoder_init_type: str = 'kaiming_normal',
                 output_init_type: str = 'kaiming_normal',
                 condition_init_type: str = 'kaiming_normal',
                 block_norm_type: str = 'none',
                 block_activation_type: str = 'relu',
                 block_init_type: str = 'resnet_zero_init',
                 norm_output:bool = True,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.norm_output = norm_output

        # Initial input-feature encoding layer
        self.input_encoder = nn.Linear(input_dim, hidden_dim)

        # One linear condition projection per ResNet block
        if condition_dim != 0:
            self.condition_projectors = nn.ModuleList([
                nn.Linear(condition_dim, hidden_dim) for _ in range(num_blocks)
            ])

        # ResNet residual blocks with shared configuration
        self.resnet_blocks = nn.ModuleList([
            ResnetBlockFC(
                input_dim=hidden_dim,
                norm_type=block_norm_type,
                activation_type=block_activation_type,
                init_type=block_init_type
            ) for _ in range(num_blocks)
        ])

        # Final output layer
        self.output_head = nn.Linear(hidden_dim, output_dim)

        # Activation before the output layer
        self.output_activation = self._get_activation_function(output_activation_type)

        # Condition fusion operator: addition or multiplication
        if condition_operator == 'add':
            self.condition_fusion = torch.add
        elif condition_operator == 'mul':
            self.condition_fusion = torch.mul
        else:
            raise ValueError(f"Unknown condition operator: {condition_operator}. Use 'add' or 'mul'.")

        # Initialize all linear layers.
        self._initialize_linear_layer(self.input_encoder, encoder_init_type)
        self._initialize_linear_layer(self.output_head, output_init_type)
        if condition_dim != 0:
            for projector in self.condition_projectors:
                self._initialize_linear_layer(projector, condition_init_type)

    def _initialize_linear_layer(self, layer: nn.Linear, init_type: str):
        """Initialize a linear layer using the selected method.

        Args:
            layer: Linear layer to initialize.
            init_type: default leaves PyTorch initialization unchanged;
                kaiming_normal/kaiming_uniform use Kaiming initialization
                (normal is suitable for ReLU); xavier_normal/xavier_uniform
                use Xavier initialization (normal suits sigmoid/tanh);
                zero clears weights and biases; small_random samples within +/-0.0001."""
        if init_type == 'default':
            return
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'zero':
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif init_type == 'small_random':
            nn.init.uniform_(layer.weight, -0.0001, 0.0001)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}. "
                             f"Use 'default', 'kaiming_normal', 'kaiming_uniform', "
                             f"'xavier_normal', 'xavier_uniform', 'zero', or 'small_random'.")

    def _get_activation_function(self, activation_type: str):
        """Return the activation function selected by activation_type."""
        activation_type = activation_type.lower()

        if activation_type == 'relu':
            return F.relu
        elif activation_type == 'leaky_relu':
            return lambda x: F.leaky_relu(x, 0.2)
        elif activation_type == 'gelu':
            return F.gelu
        elif activation_type == 'silu':
            return F.silu
        elif activation_type == 'none':
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation type: {activation_type}. "
                             f"Use 'relu', 'leaky_relu', 'gelu', 'silu', or 'none'.")

    def forward(self, inputs: torch.Tensor, condition_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the forward pass.

        Args:
            inputs: Features (B, N, input_dim) or (B, input_dim).
            condition_features: Conditions (B, N, condition_dim) or (B, condition_dim).
        Returns:
            Predictions shaped (B, N) or (B,) for a scalar output."""
        inputs = inputs.float()

        # Encode input features.
        hidden_features = self.input_encoder(inputs)

        # Apply condition modulation and a ResNet residual block at each step.
        for block_idx in range(self.num_blocks):
            # Modulate when conditioning features are available.
            if self.condition_dim != 0:
                condition_projected = self.condition_projectors[block_idx](condition_features)
                hidden_features = self.condition_fusion(hidden_features, condition_projected)

            # Apply the ResNet block.
            hidden_features = self.resnet_blocks[block_idx](hidden_features)

        # Apply the final activation and output layer.
        output = self.output_head(self.output_activation(hidden_features))

        # Squeeze the last dimension for scalar outputs.
        output = output.squeeze(-1)

        return output


# Usage examples
if __name__ == "__main__":
    # Example 1: default configuration with PyTorch initialization
    model_default = SerialModulatorShallow(
        input_dim=3,
        condition_dim=128,
        hidden_dim=256,
        num_blocks=5,
        output_dim=1,
        condition_operator='add',
        output_activation_type='relu',
        encoder_init_type='default',
        output_init_type='default',
        condition_init_type='default',
        block_norm_type='none',
        block_activation_type='relu',
        block_init_type='resnet_zero_init'
    )
    # Print the parameter count.
    total_params = sum(p.numel() for p in model_default.parameters())
    print(f"Total parameters: {total_params:,}")

    # Example 2: Kaiming initialization for ReLU networks
    model_kaiming = SerialModulatorShallow(
        input_dim=3,
        condition_dim=128,
        hidden_dim=256,
        num_blocks=5,
        output_dim=1,
        condition_operator='add',
        output_activation_type='relu',
        encoder_init_type='kaiming_normal',
        output_init_type='kaiming_normal',
        condition_init_type='kaiming_normal',
        block_norm_type='none',
        block_activation_type='relu',
        block_init_type='kaiming_normal'
    )

    # Example 3: small initialization for SDF tasks
    model_sdf = SerialModulatorShallow(
        input_dim=3,
        condition_dim=128,
        hidden_dim=256,
        num_blocks=5,
        output_dim=1,
        condition_operator='add',
        output_activation_type='relu',
        encoder_init_type='kaiming_normal',
        output_init_type='small_random',  # Use small initialization for the output layer.
        condition_init_type='xavier_uniform',
        block_norm_type='layernorm',
        block_activation_type='relu',
        block_init_type='resnet_zero_init'
    )

    # Create mock inputs.
    batch_size = 4
    num_points = 1024
    inputs = torch.randn(batch_size, num_points, 3)
    condition_features = torch.randn(batch_size, num_points, 128)

    # Run the forward pass.
    output1 = model_default(inputs, condition_features)
    output2 = model_kaiming(inputs, condition_features)
    output3 = model_sdf(inputs, condition_features)

    print(f"输入形状: inputs {inputs.shape}, condition {condition_features.shape}")
    print(f"默认模型输出形状: {output1.shape}")
    print(f"Kaiming 模型输出形状: {output2.shape}")
    print(f"SDF 模型输出形状: {output3.shape}")

    # Inspect initialization results.
    print(f"\n输出层权重统计:")
    print(
        f"默认模型 - mean: {model_default.output_head.weight.mean():.6f}, std: {model_default.output_head.weight.std():.6f}")
    print(
        f"Kaiming 模型 - mean: {model_kaiming.output_head.weight.mean():.6f}, std: {model_kaiming.output_head.weight.std():.6f}")
    print(f"SDF 模型 - mean: {model_sdf.output_head.weight.mean():.6f}, std: {model_sdf.output_head.weight.std():.6f}")
