import os

def main():
    # all params, except patient_type and day, are not used, for this is the no_lstm control type

    incSet = ['1.2']
    decSet = ['0.8']
    arg_past_step_num_severe_low_list = [5]
    correction_bolus_parameter_list = [1]
    arg_correction_bolus_ahead_step_list = [5]
    arg_severe_high_basal_list = [1.5]

    for incFactor in incSet:
        for decFactor in decSet:
            arg_patient = 'child'
            arg_day = 7
            arg_cuda = 1
            arg_consecutive_step = 2 # no use
            arg_violate_threshold = 0
            arg_last_low_basal = 0.0 # no use
            arg_last_high_basal = 1.0 # no use
            arg_low_basal = decFactor
            arg_high_basal = incFactor
            arg_max_bolus_amount_patient_type_dict = {'adult':17, 'child': 5, 'adolescent': 10}
            arg_max_bolus_amount_patient_type = arg_max_bolus_amount_patient_type_dict[arg_patient]
            arg_past_step_num_correction_bolus = 0 # no use
            for arg_past_step_num_severe_low in arg_past_step_num_severe_low_list:
                for correction_bolus_parameter in correction_bolus_parameter_list:
                    for arg_correction_bolus_ahead_step in arg_correction_bolus_ahead_step_list:
                        for arg_severe_high_basal in arg_severe_high_basal_list:
                            arg_id = '{correction_bolus_parameter}_{arg_correction_bolus_ahead_step}_{arg_severe_high_basal}'.format(correction_bolus_parameter=correction_bolus_parameter,
                                                                                arg_correction_bolus_ahead_step=arg_correction_bolus_ahead_step,arg_severe_high_basal=arg_severe_high_basal)
                            cline = "python medical_case_pipeline_new_controller_structure_no_lstm.py --arg_id {arg_id}  --arg_cuda {arg_cuda}  --arg_patient {arg_patient}  --arg_day {arg_day}  " \
                                    "--arg_consecutive_step {arg_consecutive_step} --arg_violate_threshold {arg_violate_threshold}  --arg_last_low_basal {arg_last_low_basal} " \
                                    "--arg_last_high_basal {arg_last_high_basal} --arg_low_basal {arg_low_basal} " \
                                    "--arg_high_basal {arg_high_basal} --correction_bolus_parameter {correction_bolus_parameter} " \
                                    "--arg_max_bolus_amount_patient_type {arg_max_bolus_amount_patient_type} " \
                                    "--arg_past_step_num_severe_low {arg_past_step_num_severe_low} " \
                                    "--arg_past_step_num_correction_bolus {arg_past_step_num_correction_bolus} " \
                                    "--arg_correction_bolus_ahead_step {arg_correction_bolus_ahead_step} " \
                                    "--arg_severe_high_basal {arg_severe_high_basal}".format(arg_id=arg_id, arg_cuda=arg_cuda, arg_patient=arg_patient,
                                                                    arg_day=arg_day, arg_consecutive_step=arg_consecutive_step, arg_violate_threshold=arg_violate_threshold,
                                                                    arg_last_low_basal=arg_last_low_basal, arg_last_high_basal=arg_last_high_basal,
                                                                    arg_low_basal=arg_low_basal, arg_high_basal=arg_high_basal,
                                                                    correction_bolus_parameter=correction_bolus_parameter,
                                                                    arg_max_bolus_amount_patient_type=arg_max_bolus_amount_patient_type,
                                                                    arg_past_step_num_severe_low=arg_past_step_num_severe_low,
                                                                    arg_past_step_num_correction_bolus=arg_past_step_num_correction_bolus,
                                                                    arg_correction_bolus_ahead_step=arg_correction_bolus_ahead_step,
                                                                     arg_severe_high_basal=arg_severe_high_basal)
                            print(cline)
                            os.system(cline)
    



            

if __name__ == "__main__":
    main()
