import shutil
import os
import pandas as pd
import numpy as np 
import subprocess
from pyflowmeter.sniffer import create_sniffer


def capToCicids2018windows(file_name, index = False,header=True ):
    """
    file_name: String -> Nom du fichier .pcap_flow.csv dans c/users/bapti/desktop/CICFlowmeter/flows
    index: Bool = False -> Permet de chosir de laisser ou non les index dans le fichier output
    hearders: Bool = True ->  Permet de chosir de laisser ou non les headers dans le fichier output

    On va importer le fichier flows obtenue par CICFlowmeter, puis on va le convertir au format Cidids2018. 
    """
    
    # Chemin du répertoire de destination et de sortie dans WSL
    wsl_destination_path = 'import'
    wsl_output_path = 'output_Cicid'

    windows_source_path = '/mnt/c/users/bapti/desktop/CICFlowmeter/flows'

    # Construction des chemins complets  
    windows_file_path = os.path.join(windows_source_path, file_name)
    wsl_destination_file_path = os.path.join(wsl_destination_path, file_name)
    wsl_output_file_path = os.path.join(wsl_output_path, f"{file_name.replace('.csv', '_processed.csv')}")
    
    print(f'Windows File Path: {windows_file_path}')
    print(f'WSL Destination Path: {wsl_destination_path}')

    # Copie Windows --> WSL
    shutil.copy(windows_file_path, wsl_destination_file_path)

    # Traitement du DataFrame
    df_CICFlow = pd.read_csv(wsl_destination_file_path)
    df_CICFlow = df_CICFlow.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP'])
    df_CICFlow.to_csv(wsl_output_file_path, index=index,header=header)

    print(f'Fichier traité et copié avec succès dans {wsl_output_file_path}')


def capToCicids2018Linux(file_name, index = False,header=True ):
    """
    file_name: String -> Nom du fichier .pcap_flow.csv dans c/users/bapti/desktop/CICFlowmeter/flows
    index: Bool = False -> Permet de chosir de laisser ou non les index dans le fichier output
    hearders: Bool = True ->  Permet de chosir de laisser ou non les headers dans le fichier output

    On va importer le fichier flows obtenue par CICFlowmeter, puis on va le convertir au format Cidids2018. 
    """
    if not os.path.isfile(file_name):
        print(f"Le fichier {file_name} n'existe pas.")
        return
    
    output_folder = "output_Cicids"
    file_output = os.path.join(output_folder, os.path.basename(file_name).replace(".pcap", ".csv"))
    print(f"{file_output=}")


    sniffer = create_sniffer(
            input_file=file_name,
            to_csv=True,
            output_file=file_output,
        )

    sniffer.start()
    try:
        sniffer.join()
    except KeyboardInterrupt:
        print('Stopping the sniffer')
        sniffer.stop()
    finally:
        sniffer.join()

    
    

    # Traitement du DataFrame
    df_CICFlow = pd.read_csv(file_output)
    df_CICFlow = df_CICFlow.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP'])
    df_CICFlow.to_csv(file_output, index=index,header=header)

    print(f'Fichier traité et copié avec succès dans {file_output}')



def test_compare_column_names_Cicids():
    """
    Compare les noms de colonnes de deux DataFrames et affiche les résultats.
    """

    df_a_test = pd.read_csv('output_Cicids/test.pcap_Flow_processed.csv')
    df_CICFlow = pd.read_csv('Datas/CICIDS2018_1.csv')

    cols1 = df_a_test.columns
    cols2 = df_CICFlow.columns
    print(np.all(cols1 == cols2))
    assert np.all(cols1 == cols2)


def capToHikari2021(file_name, window_file = False):
    #On va chercher la capture et on la copie dans un dossier cap

    wsl_destination_path = 'captures'
    wsl_log_path = 'log_Hiraki'
    wsl_output_path = 'output_Hiraki'
    if window_file:
        windows_source_path = '/mnt/c/users/bapti/desktop/Captures'
 
        windows_file_path = os.path.join(windows_source_path, file_name)
        wsl_destination_file_path = os.path.join(wsl_destination_path, file_name)

        shutil.copy(windows_file_path, wsl_destination_file_path)

    #On la convertie en log 'sudo /opt/zeek/bin/zeek flowmeter -r cap_test.pcapng'

    command = "sudo /opt/zeek/bin/zeek flowmeter -r captures/" + file_name 

    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")  

    #On récupère le log, on le met dans un fichier log + on récup le conn.log ! 

    wsl_log_file_path = os.path.join(wsl_log_path, f"{file_name.replace('.pcap', '_conn.log')}")
    
    shutil.copy("conn.log", wsl_log_file_path)
    ##CONVERTION CONN.LOG -> CONN.CSV
    wsl_output_file_path = os.path.join(wsl_output_path, f"{file_name.replace('.pcap', '_conn.csv')}")
    
    command = 'awk \'BEGIN{OFS=",";}{print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100}\' ' +  wsl_log_file_path + ' > ' + wsl_output_file_path
    
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
        # On retire les lignes 1 à 5  + 7 + la dernière 
    # Supprimer les lignes 1 à 5 et la ligne 7
    wsl_output_temp_file_path = os.path.join(wsl_output_path, f"{file_name.replace('.pcap', 'conn_temp.csv')}")

    command ="sed -e '1,5d' -e '8d' " + wsl_output_file_path + " > " +  wsl_output_temp_file_path
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        

    # Obtenir le nombre total de lignes dans le fichier_temp.csv
    command = "total_lines=$(wc -l <" +  wsl_output_temp_file_path + ")"
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # Supprimer la dernière ligne
    command = "head -n $(($total_lines - 1)) " + wsl_output_temp_file_path +" > " + wsl_output_file_path
    
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # Supprimer le fichier temporaire
    command = "rm " + wsl_output_temp_file_path
    try:
        subprocess.run(command, shell=True, check=True, text=True)
        print(f'Le fichier {wsl_output_temp_file_path} a bien été générer sans encombre') 
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
    ##CONVERTION .LOG -> .CSV
    wsl_log_file_path = os.path.join(wsl_log_path, f"{file_name.replace('.pcap', '.log')}")
    
    shutil.copy("flowmeter.log", wsl_log_file_path)

    
    

    #On le convertie en en csv awk 'BEGIN{OFS=",";}{print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100}' flowmeter.log > flowmeter.csv
    wsl_output_file_path = os.path.join(wsl_output_path, f"{file_name.replace('.pcap', '.csv')}")
    
    command = 'awk \'BEGIN{OFS=",";}{print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100}\' ' +  wsl_log_file_path + ' > ' + wsl_output_file_path
    
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
   


    # On retire les lignes 1 à 5  + 7 + la dernière 
    # Supprimer les lignes 1 à 5 et la ligne 7
    wsl_output_temp_file_path = os.path.join(wsl_output_path, f"{file_name.replace('.pcap', '_temp.csv')}")

    command ="sed -e '1,5d' -e '8d' " + wsl_output_file_path + " > " +  wsl_output_temp_file_path
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        

    # Obtenir le nombre total de lignes dans le fichier_temp.csv
    command = "total_lines=$(wc -l <" +  wsl_output_temp_file_path + ")"
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # Supprimer la dernière ligne
    command = "head -n $(($total_lines - 1)) " + wsl_output_temp_file_path +" > " + wsl_output_file_path
    
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # Supprimer le fichier temporaire
    command = "rm " + wsl_output_temp_file_path
    try:
        subprocess.run(command, shell=True, check=True, text=True)
        print(f'Le fichier {wsl_output_temp_file_path} a bien été générer sans encombre') 
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


    #On retire éventuellement les colonnes inutiles 

    # Supprimer tout les fichiers .log du répertoire courrant  rm *.txt
    command = "rm *.log -f"
    
    try:
        subprocess.run(command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    
    




if __name__ == "__main__":
    #test_compare_column_names_Cicids("test.pcap_Flow.csv")
    pass

