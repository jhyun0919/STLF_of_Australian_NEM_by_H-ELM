%TrainingData_File='csv_data/ACTEWAGL/train_wd.csv';TestingData_File='csv_data/ACTEWAGL/test_wd.csv'; FileName='input_data/ACTEWAGL/elm_input_wd';
%TrainingData_File='csv_data/ACTEWAGL/train_lwd.csv';TestingData_File='csv_data/ACTEWAGL/test_lwd.csv'; FileName='input_data/ACTEWAGL/elm_input_lwd';
%TrainingData_File='csv_data/ACTEWAGL/train_lww.csv';TestingData_File='csv_data/ACTEWAGL/test_lww.csv'; FileName='input_data/ACTEWAGL/elm_input_lww';
TrainingData_File='csv_data/ACTEWAGL/train_lwdw.csv';TestingData_File='csv_data/ACTEWAGL/test_lwdw.csv'; FileName='input_data/ACTEWAGL/elm_input_lwdw';


No_of_Output = 48;
elm_parser(TrainingData_File, TestingData_File , No_of_Output, FileName);