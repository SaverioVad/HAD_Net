import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, k_in_init, k_out_init):
        super(Discriminator, self).__init__()

        def disc_block(k_in, k_out, inst_norm):
            
            if(inst_norm == True):
                layers = nn.Sequential(
                            nn.Conv3d(in_channels=k_in, out_channels=k_out, kernel_size=4, stride=2, padding=1),
                            nn.InstanceNorm3d(k_out),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True)
                        )
            else:
                layers = nn.Sequential(
                            nn.Conv3d(in_channels=k_in, out_channels=k_out, kernel_size=4, stride=2, padding=1),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True)
                        )
            return layers

        self.disc_1 = disc_block(k_in_init, 4*k_out_init, False) # 128
        self.disc_2 = disc_block(8*k_out_init, 8*k_out_init, True) # 256
        self.disc_3 = disc_block(16*k_out_init, 16*k_out_init, True) # 512
        self.disc_4 = disc_block(32*k_out_init, 32*k_out_init, True) # 1024
        self.disc_5 = nn.Conv3d(in_channels=32*k_out_init, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False)

    def forward(self, segmap, mods, feature_maps):
        
        # Concatenate segmentaiton map with the input modalities
        x = torch.cat((segmap, mods), dim=1)
        
        # Pass x through the first discriminator block
        x = self.disc_1(x)
        
        # Extract feature maps for this scale. Upscale the first set of feature maps to make
        # sure they are the same size. Concat with output of those scale and pass through disc block.
        fmap_1 = feature_maps[0][0]
        fmap_2 = feature_maps[0][1]
        fmap_2 = F.interpolate(fmap_2, size=fmap_1.size()[-3:])
        features = torch.cat((fmap_1, fmap_2), dim=1)
        x = torch.cat((x, features), dim=1)        
        x = self.disc_2(x)
        
        # Repeat process for next block.
        fmap_1 = feature_maps[1][0]
        fmap_2 = feature_maps[1][1]
        fmap_2 = F.interpolate(fmap_2, size=fmap_1.size()[-3:])
        features = torch.cat((fmap_1, fmap_2), dim=1)
        x = torch.cat((x, features), dim=1)
        x = self.disc_3(x)
        
        # Repeat process for next block.
        fmap_1 = feature_maps[2][0]
        fmap_2 = feature_maps[2][1]
        fmap_2 = F.interpolate(fmap_2, size=fmap_1.size()[-3:])
        features = torch.cat((fmap_1, fmap_2), dim=1)
        x = torch.cat((x, features), dim=1)
        x = self.disc_4(x)
        
        # Pass through final block.
        x = self.disc_5(x)
        return x