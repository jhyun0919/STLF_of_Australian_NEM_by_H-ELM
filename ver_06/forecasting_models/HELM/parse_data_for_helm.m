%TrainingData_File='csv_data/ACTEWAGL/train_wd.csv';TestingData_File='csv_data/ACTEWAGL/test_wd.csv'; FileName='input_data/ACTEWAGL/helm_input_wd';
%TrainingData_File='csv_data/ACTEWAGL/train_lwd.csv';TestingData_File='csv_data/ACTEWAGL/test_lwd.csv'; FileName='input_data/ACTEWAGL/helm_input_lwd';
%TrainingData_File='csv_data/ACTEWAGL/train_lww.csv';TestingData_File='csv_data/ACTEWAGL/test_lww.csv'; FileName='input_data/ACTEWAGL/helm_input_lww';
%TrainingData_File='csv_data/ACTEWAGL/train_lwdw1.csv';TestingData_File='csv_data/ACTEWAGL/test_lwdw1.csv'; FileName='input_data/ACTEWAGL/helm_input_lwdw1';
%TrainingData_File='csv_data/ACTEWAGL/train_lwdw2.csv';TestingData_File='csv_data/ACTEWAGL/test_lwdw2.csv'; FileName='input_data/ACTEWAGL/helm_input_lwdw2';
%TrainingData_File='csv_data/ACTEWAGL/train_lwdw3.csv';TestingData_File='csv_data/ACTEWAGL/test_lwdw3.csv'; FileName='input_data/ACTEWAGL/helm_input_lwdw3';
%TrainingData_File='csv_data/ACTEWAGL/train_lwdw4.csv';TestingData_File='csv_data/ACTEWAGL/test_lwdw4.csv'; FileName='input_data/ACTEWAGL/helm_input_lwdw4';
%TrainingData_File='csv_data/ACTEWAGL/train_lwdw5.csv';TestingData_File='csv_data/ACTEWAGL/test_lwdw5.csv'; FileName='input_data/ACTEWAGL/helm_input_lwdw5';
%TrainingData_File='csv_data/ACTEWAGL/train_lwdw6.csv';TestingData_File='csv_data/ACTEWAGL/test_lwdw6.csv'; FileName='input_data/ACTEWAGL/helm_input_lwdw6';

No_of_Output = 48;
helm_parser(TrainingData_File, TestingData_File , No_of_Output, FileName);