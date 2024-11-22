# GANs
Code about GANs



graph TD
    subgraph Generator
    Z[Noise z ∈ R^128] --> FC[FC 128->8192]
    FC --> BN1[BatchNorm]
    BN1 --> ReLU1[ReLU]
    ReLU1 --> Reshape[Reshape to 512×4×4]
    
    Reshape --> ResBlock1[ResBlock 512->512]
    ResBlock1 --> ResBlock2[ResBlock 512->256]
    ResBlock2 --> ResBlock3[ResBlock 256->128]
    ResBlock3 --> ResBlock4[ResBlock 128->64]
    
    ResBlock4 --> ReLU2[ReLU]
    ReLU2 --> Conv[Conv 64->3]
    Conv --> Tanh[Tanh]
    Tanh --> Output[Output 3×64×64]
    
    subgraph "ResBlock Structure"
        direction LR
        In1[Input] --> Conv1_1[Conv 3×3]
        Conv1_1 --> BN1_1[BatchNorm]
        BN1_1 --> ReLU1_1[ReLU]
        ReLU1_1 --> Conv1_2[Conv 3×3]
        Conv1_2 --> BN1_2[BatchNorm]
        BN1_2 --> ReLU1_2[ReLU]
        ReLU1_2 --> Upsample1[Upsample ×2]
        
        In1 --> Bypass1[Upsample ×2]
        Upsample1 --> Add1((+))
        Bypass1 --> Add1
    end
    end
