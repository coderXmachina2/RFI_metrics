import gzip
import shutil
import glob
import astropy
import matplotlib.pyplot as plt
import numpy as np
import astropy
import math
import re
import psrchive
import tensorflow as tf

from copy import deepcopy as cp
from scipy import stats
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

plt.rcParams["figure.figsize"] = (4,4)

def tensorflow_metrics(ag_labs, nn_predictions):
    area_under_roc_append = []
    area_under_pr_curve = []    

    for l in range(0, len(nn_predictions)): 
        for t in range(0,len(ag_labs)): #
            if(nn_predictions[l][0][19:29]+"_"+nn_predictions[l][0][57:-3] == ag_labs[t][99:-4]):
                print("Evidence of crossmatch")
                print("l:", l, "t:", t)
                data_label = np.load(ag_labs[t])
                print("Data:", nn_predictions[l][0])
                
                skl_fpr, skl_tpr, thresholds = metrics.roc_curve(data_label, nn_predictions[l][1], pos_label=1)
                skl_precision, skl_recall, skl_thresholds = precision_recall_curve(data_label, nn_predictions[l][1])
                
                plt.plot(skl_recall, skl_precision)
                plt.title('sklearn RR curve')
                plt.ylabel('Precision')
                plt.xlabel('Recall')
                plt.grid()
                plt.show()
                
                plt.plot(skl_fpr, skl_tpr)
                plt.plot([0,1], [0,1] )
                plt.title('sklearn ROC curve')
                plt.ylabel('True Positive Rate (TPR/Sensitivity/ Recall)')
                plt.xlabel('False Positive Rate (FPR/1-Specificity)')
                plt.grid()
                plt.show()
                
                area_under_pr_curve.append(metrics.auc(skl_recall, skl_precision))
                area_under_roc_append.append(metrics.auc(skl_fpr, skl_tpr))
                
                print("AUPRC:", metrics.auc(skl_recall, skl_precision))
                print("AUROC:", metrics.auc(skl_fpr, skl_tpr))
    
    return area_under_roc_append, area_under_pr_curve

def make_AUROC(ag_labs, nn_predictions):
    my_area_under_roc_append = []
    my_area_under_pr_curve = []    

    for l in range(0, len(nn_predictions)):
        for t in range(0,len(ag_labs)):
            if(nn_predictions[l][0][19:29]+"_"+nn_predictions[l][0][57:-3] == ag_labs[t][99:-4]):
                archive_2 = psrchive.Archive_load(nn_predictions[l][0])
                subint_2 = archive_2.get_data()

                data_label = np.load(ag_labs[t])
                print("Data:", nn_predictions[l][0])

                sense_points_on_the_ROC = []
                FPR_points_on_the_ROC = []
                precisions = []
                recall_smaller_set = []
                f1_score_arr = []
                
                trial_thresh = np.linspace(0, 1, num = 500)#, num=100)

                for threshes in trial_thresh:
                    template = cp(nn_predictions[l][1])

                    for r in range(0, len(template)):
                        if(template[r] < threshes): #If its less than the threshold it becomes 0
                            template[r] = 0
                        else:
                            template[r] = 1

                    NTP = 0
                    NTN = 0
                    NFP = 0
                    NFN = 0

                    for v in range(0, len(data_label)):
                        if(data_label[v] == 0 and template[v] == 0):
                            NTP+=1
                        elif(data_label[v] == 0 and template[v] == 1):
                            NFN+=1
                        elif(data_label[v] == 1 and template[v] == 0):
                            NFP+=1
                        elif(data_label[v] == 1 and template[v] == 1):
                            NTN+=1
                            
                    sense = (NTP)/(NTP+NFN) #Recall
                    speci = (NTN)/(NTN+NFP)
                    my_fpr = 1 - speci
                    
                    if((NTP + NFP) != 0):
                        precision = (NTP)/(NTP+NFP)
                        precisions.append(precision)
                        recall_smaller_set.append(sense)

                    sense_points_on_the_ROC.append(sense)
                    FPR_points_on_the_ROC.append(my_fpr)
                
                plt.plot(recall_smaller_set, precisions ) #x is recall, y is precision
                plt.title('My PR Curve') 
                plt.ylabel('Precision')
                plt.xlabel('Recall')
                plt.grid()
                plt.show()
                    
                plt.plot(FPR_points_on_the_ROC, sense_points_on_the_ROC  ) #x and y
                plt.plot([0,1], [0,1] )
                plt.title('My Receiver Operating Curve')
                plt.ylabel('True Positive Rate (TPR/Sensitivity/ Recall)')
                plt.xlabel('False Positive Rate (FPR/1-Specificity)')
                plt.grid()
                plt.show()

                print("AUPR calculated from my array", metrics.auc(recall_smaller_set, precisions  ))
                print("AUROC calculated from my array", 1-metrics.auc( sense_points_on_the_ROC , FPR_points_on_the_ROC))

                my_area_under_roc_append.append(  1-metrics.auc( sense_points_on_the_ROC , FPR_points_on_the_ROC)       )
                my_area_under_pr_curve.append(metrics.auc(recall_smaller_set, precisions))
                
    return my_area_under_roc_append, my_area_under_pr_curve

def try_again(true_labs, predictions, plot_verbose):
    print("Analytics is awesome!")
    print("True Labels:", true_labs)
    print("Predictions:", predictions)
    
    ROC_sense = []
    ROC_FPR = []
    precisions = []
    smaller_sense = []
    f1_score = []
    thresh_steps = 80
    
    print("Threshold steps:", thresh_steps)
    threshes = np.linspace(0, 1, num = thresh_steps)#, num=100)

    for trial_thresh in threshes:
        template = cp(predictions)

        for r in range(0, len(template)): #Binarize
            if(template[r] <= trial_thresh): #If its less than the threshold it becomes 0
                template[r] = 0
            else:
                template[r] = 1
                
        #plt.plot( template )  #blue
        #plt.plot( true_labs ) #orange
        #plt.show()

        NTP = 0
        NTN = 0
        NFP = 0
        NFN = 0
        #print("True label, predicted")
        for v in range(0, len(true_labs)):
            if(true_labs[v] == 0 and template[v] == 0): #predicted 0 when its actually 0
                NTP+=1
                #print(true_labs[v], template[v] , "True Positive"   )
            elif(true_labs[v] == 0 and template[v] == 1): #predicted 1 when its actually 0
                NFN+=1
                #print(true_labs[v], template[v] , "False Negative"   )
            elif(true_labs[v] == 1 and template[v] == 0): #predicted 0 when its actually 1
                NFP+=1
                #print(true_labs[v], template[v] , "False Positive"   )
            elif(true_labs[v] == 1 and template[v] == 1): #predicted 1 when its actually 1
                NTN+=1
                #print(true_labs[v], template[v] , "True Negative"   )
                
        tn, fp, fn, tp = confusion_matrix(true_labs, template, labels=[1,0]).ravel()
            
        cm = confusion_matrix(true_labs, template)
        
        #I accept that these are correct
        #print(NTN+ NFP+ NFN+ NTP  )
        #print("TN, FP, FN, TP")
        #print("My matrix:", NTN, NFP, NFN, NTP)
        #print("SK matrix:", tn, fp, fn, tp)
        #print("Confusion matrix:\n", cm, "\n") #In our case the majority will be the True Negative. Clean channels
            
        sense = (NTP)/(NTP+NFN) #Recall
        speci = (NTN)/(NTN+NFP)
        my_fpr = 1 - speci
                    
        if((NTP + NFP) != 0):
            precision = (NTP)/(NTP+NFP)
            
            precisions.append(precision)
            smaller_sense.append(sense)
            if((precision + sense) == 0):
                f1 = 0
            else:
                f1 = 2 * (precision * sense) / (precision + sense)
            f1_score.append(f1)
            
        ROC_sense.append(sense)
        ROC_FPR.append(my_fpr)
        
    if(plot_verbose):
        plt.plot(ROC_FPR, ROC_sense)
        plt.plot([0,1], [0,1] )
        plt.title('sklearn ROC curve')
        plt.ylabel('True Positive Rate (TPR aka Sensitivity aka Recall)')
        plt.xlabel('False Positive Rate (FPR aka 1-Specificity)')
        plt.grid()
        plt.show()

        plt.plot(smaller_sense, precisions ) #x is recall, y is precision
        plt.title('My PR Curve') 
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.grid()
        plt.show()

        plt.plot(precisions)
        plt.plot(smaller_sense)
        plt.plot(f1_score)
        plt.title("Precision, recall, F1")
        plt.grid()
        plt.show()
    
    print("AUROC:", metrics.auc(ROC_FPR, ROC_sense))
    print("AUPRC:", metrics.auc(smaller_sense, precisions))
    print("Max F1:", np.max(f1_score))
    
    return( metrics.auc(ROC_FPR, ROC_sense),metrics.auc(smaller_sense, precisions),  np.max(f1_score)   )