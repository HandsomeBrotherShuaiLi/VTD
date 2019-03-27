from Recognition.CRNN.main import Recognition
app=Recognition(
    img_dir='D:\python_projects\data_new\data\\train_img_cut_new',
    labeled_file_path='D:\python_projects\data_new\data\\train_img_labels\labels.txt',
    chinese_set_path='D:\python_projects\data_new\data\chinese\chinese_all.txt',
    first_use=False,text_max_len=35,version='v2',fixed_size=False,train_batch_size=1
)
app.train()