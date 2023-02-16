from pymlpipe.api import Client

ml_connect=Client()
print(ml_connect.get_all_experiments())
print(ml_connect.get_all_run_ids("IrisAutoML"))
#print(ml_connect.get_run_details("Pytorch","01d9d974-284c-4775-95bc-792491267d05"))
#print(ml_connect.get_all_run_details("IrisAutoML"))
#print(ml_connect.get_metrics_comparison("Pytorch",format="pandas",sort_by="f1"))
print(ml_connect.get_model_details("IrisAutoML","680f5dcf-e207-4cb5-adb9-cc6d7fbb8b16",format="pandas"))
