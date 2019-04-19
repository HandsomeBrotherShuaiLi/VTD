from Recognition.CRNN.main import Recognition
app=Recognition(
    img_dir='D:\py_projects\data_new\data_new\data\\train_img_cut_new',
    labeled_file_path='D:\py_projects\data_new\data_new\data\\train_img_labels',
    chinese_set_path='D:\py_projects\data_new\data_new\data\chinese\chinese_all.txt',
    first_use=False,text_max_len=35,version='v1',fixed_size=True,train_batch_size=50
)
app.predict(predict_model_version='v3',mode=1,
            model_path='D:\py_projects\VTD\model\crnn_v3_fixed_size_True_isGRU_False\crnn--73--1.0413--0.87700.hdf5')