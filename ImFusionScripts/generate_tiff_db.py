import os

db_path = "D:\\Maria\\DataBases\\SpineIFL\\FullRawDb"
output_path = "D:\\IlkerDb"


name_list = os.listdir(db_path)
acquisitions_list = ["spinous_process", "vertebrae_convex", "vertebrae_linear"]

file_lines = ["INPUTFILE;", "OUTPUTFOLDER"]


for subject_id, subject in enumerate(name_list):

    updated_acquisition_list = [item for item in os.listdir(os.path.join(db_path, subject)) if acquisitions_list[0] in item
                                or acquisitions_list[1] in item or acquisitions_list[2] in item]

    print(updated_acquisition_list, " - ", subject)

    for acquisition in updated_acquisition_list:

        data_path = os.path.join(db_path, subject, acquisition)

        filename = [item for item in os.listdir(data_path) if ".imf" in item]

        filepath = os.path.join(data_path, filename[0])
        out_folder = os.path.join(output_path, "sub" + str(subject_id) + acquisition)

        os.mkdir(out_folder)

        current_line = "\n" + filepath + ";" + out_folder
        file_lines.append(current_line)


with open("tmp.txt", 'w') as fid:
    fid.writelines(file_lines)