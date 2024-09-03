import torch
import torch.nn as nn

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=(3,224,224), patch_size=(3,16,16), in_c=1, embed_dim=768):
        super(PatchEmbed3D, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dhw = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop)
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class VisionTransformer3D(nn.Module):
    def __init__(self, img_size=(3,224,224), patch_size=(3,16,16), in_c=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.):
        super(VisionTransformer3D, self).__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_c, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder part with U-Net like structure
        self.conv_before_up = nn.Conv3d(embed_dim, 512, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=(3,2,2), stride=(3,2,2))
        self.decoder4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(1,2,2), stride=(1,2,2))
        self.decoder3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=(1,2,2), stride=(1,2,2))
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Reshape and project to match decoder input dimensions
        B, N, C = x.shape
        D, H, W = self.patch_embed.dhw
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        x = self.conv_before_up(x)  # Project to suitable dimension

        # Decoder
        x = self.upconv4(x)
        x = self.decoder4(x)

        x = self.upconv3(x)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = self.decoder1(x)

        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 2, 3, 224, 224)
    model = VisionTransformer3D(img_size=(3,224,224), patch_size=(3,16,16), in_c=2, embed_dim=768, depth=16, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.)
    print(model(x).shape)