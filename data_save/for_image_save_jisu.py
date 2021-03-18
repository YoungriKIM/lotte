# 지수가 만들어준 포문으롤 이미지 저장하기

def make_file_list():
    
    train_img_list = list()

    for img_idx in range(200):
        img_path = "./Your/data_1/path" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./Your/data_2/path" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list