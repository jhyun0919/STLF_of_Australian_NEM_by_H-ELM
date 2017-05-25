%TrainingData_File='csv_data/train_d.csv';TestingData_File='csv_data/test_d.csv'; FileName='input_data/helm_input_d';
%TrainingData_File='csv_data/train_w.csv';TestingData_File='csv_data/test_w.csv'; FileName='input_data/helm_input_w';
%TrainingData_File='csv_data/train_dw.csv';TestingData_File='csv_data/test_dw.csv'; FileName='input_data/helm_input_dw';
TrainingData_File='csv_data/train.csv';TestingData_File='csv_data/test.csv'; FileName='input_data/helm_input';

No_of_Output = 48;
helm_parser(TrainingData_File, TestingData_File , No_of_Output, FileName);