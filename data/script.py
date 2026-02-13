import os 

if __name__ == "__main__":
    data_dir = "./vimeo-500/masks"

    # with open("./vimeo-500/train.txt", 'w') as f:
    #     f.writelines([line + '\n' for line in os.listdir(data_dir)])

    with open("./vimeo-500/train.txt", 'r') as f:
        postfix = []
        lines = f.readlines()
        for seq in lines:
            seq = seq.strip()
            mask_path = os.path.join('vimeo-500', 'masks', seq)
            
            if os.path.exists(mask_path):
                postfix.extend([f"{seq}/{imgs}" for imgs in os.listdir(mask_path)])
            
        print(postfix)