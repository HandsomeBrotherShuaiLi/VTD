from Recognition.CRNN.main import Recognition
app=Recognition(
    img_dir='D:\py_projects\data_new\data_new\data\\train_img_cut_new',
    labeled_file_path='D:\py_projects\data_new\data_new\data\\train_img_labels',
    chinese_set_path='D:\py_projects\data_new\data_new\data\chinese\chinese_all.txt',
    first_use=False,text_max_len=35,version='v1',fixed_size=True,train_batch_size=50
)
app.train()
# app.predict(predict_model_version='v1',mode=1,
#             model_path='model/crnn_v1_fixed_size_True_isGRU_False/crnn_--39--1.070.hdf5')