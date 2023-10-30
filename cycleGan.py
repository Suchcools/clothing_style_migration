import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from units import CycleGANDataset,Mask_replace
from models.generator import Generator
from models.discriminator import Discriminator
from tqdm import tqdm
# 在这个示例中，我们假设已经定义了 `Generator` 和 `Discriminator` 类，以及数据集类 `CycleGANDataset`，并将它们导入到代码中。我们定义了 `gen_A_to_B` 和 `gen_B_to_A` 作为我们的生成器，并定义 `disc_A` 和 `disc_B` 作为我们的判别器。我们使用 `MSELoss` 作为循环一致性损失，使用 `BCEWithLogitsLoss` 作为GAN损失。我们还定义了三个优化器，分别用于优化生成器、判别器A和判别器B。

# 在训练循环中，我们首先从数据集中获取一对图像 `A` 和 `B`，并将它们送入模型。我们使用生成器将 `A` 转换为 `B`，并使用另一个生成器将 `B` 转换为 `A`。我们计算生成器的总损失，包括对抗损失、循环一致性损失和身份损失。我们还分别训练两个判别器，将其区分真实图像和生成图像。最后，我们保存模型和生成的图像。

# 在训练完成后，我们可以使用训练好的模型生成新的图像。我们加载训练好的
device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. 定义生成器和判别器的架构
gen_A_to_B = Generator().to(device)
gen_B_to_A = Generator().to(device)
disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)

# 2. 定义损失函数
mse_loss = nn.MSELoss()
gan_loss = nn.BCEWithLogitsLoss()

# 3. 定义优化器
gen_optim = optim.Adam(list(gen_A_to_B.parameters()) + list(gen_B_to_A.parameters()), lr=0.0002, betas=(0.5, 0.999))
disc_A_optim = optim.Adam(disc_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_B_optim = optim.Adam(disc_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 4. 训练模型
num_epochs=3
save_path='save/'
dataset = CycleGANDataset()  # 自己定义的数据集
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
gen_adv_loss_list=[]
gen_cycle_loss_list=[]
gen_total_loss_list=[]
gen_idt_loss_list=[]
disc_A_loss_list=[]
disc_B_loss_list=[]

for epoch in range(num_epochs):
    for i, (A, B , _) in tqdm(enumerate(dataloader)):
        real_A = A.to(device).float()
        real_B = B.to(device).float()

        # 训练生成器
        gen_optim.zero_grad()
        fake_B = gen_A_to_B(real_A)
        cycle_A = gen_B_to_A(fake_B)
        fake_A = gen_B_to_A(real_B)
        cycle_B = gen_A_to_B(fake_A)
        idt_A = gen_B_to_A(real_A)
        idt_B = gen_A_to_B(real_B)
        gen_adv_loss = gan_loss(disc_B(fake_B), torch.ones_like(disc_B(fake_B))).to(device)
        gen_cycle_loss = mse_loss(cycle_A, real_A) + mse_loss(cycle_B, real_B)
        gen_idt_loss = mse_loss(idt_A, real_A) + mse_loss(idt_B, real_B)
        gen_total_loss = gen_adv_loss + 10 * gen_cycle_loss + 5 * gen_idt_loss #对抗损失和循环一致性损失 总体损失
        gen_total_loss.backward()

        gen_adv_loss_list.append(float(gen_adv_loss))
        gen_cycle_loss_list.append(float(gen_cycle_loss))
        gen_idt_loss_list.append(float(gen_total_loss))
        gen_optim.step()

        # 训练判别器 A
        disc_A_optim.zero_grad()
        real_A_loss = gan_loss(disc_A(real_A), torch.ones_like(disc_A(real_A)).to(device))
        fake_A_loss = gan_loss(disc_A(fake_A.detach()), torch.zeros_like(disc_A(fake_A)).to(device))
        disc_A_loss = real_A_loss + fake_A_loss
        disc_A_loss.backward()
        disc_A_optim.step()
        disc_A_loss_list.append(float(disc_A_loss))

        # 训练判别器 B
        disc_B_optim.zero_grad()
        real_B_loss = gan_loss(disc_B(real_B), torch.ones_like(disc_B(real_B)))
        fake_B_loss = gan_loss(disc_B(fake_B.detach()), torch.zeros_like(disc_B(fake_B)))
        disc_B_loss = real_B_loss + fake_B_loss
        disc_B_loss.backward()
        disc_B_optim.step()
        disc_B_loss_list.append(float(disc_B_loss))

    # # 每个epoch结束后，保存模型并生成图像
    if epoch % 5 == 0:
        print(f'Epoch {epoch} Over, Disc_A_loss : {round(float(disc_A_loss),2)},  Disc_B_loss : {round(float(disc_B_loss),2)}')
        torch.save(gen_A_to_B.state_dict(), f"{save_path}gen_A_to_B_{epoch}.pt")
        torch.save(gen_B_to_A.state_dict(), f"{save_path}gen_B_to_A_{epoch}.pt")
        torch.save(disc_A.state_dict(), f"{save_path}disc_A_{epoch}.pt")
        torch.save(disc_B.state_dict(), f"{save_path}disc_B_{epoch}.pt")

    with torch.no_grad():
        # 生成 A -> B 的图像
        A, _, Mask= dataset[0]
        A = torch.Tensor(A).unsqueeze(0).to(device)
        B = gen_A_to_B(A)
        save_image(B.cpu(), f"{save_path}gen_A_to_B_{epoch}.png")
        
        # 生成 B -> A 的图像
        _, B, Mask= dataset[0]
        B = torch.Tensor(B).unsqueeze(0).to(device)
        A = gen_B_to_A(B)
        save_image(A.cpu(), f"{save_path}gen_B_to_A_{epoch}.png")


#使用模型生成图像
#加载训练好的模型
gen_A_to_B = Generator()
gen_A_to_B.load_state_dict(torch.load(f"{save_path}gen_A_to_B_{epoch}.pt"))
gen_B_to_A = Generator()
gen_B_to_A.load_state_dict(torch.load(f"{save_path}gen_B_to_A_{epoch}.pt"))
