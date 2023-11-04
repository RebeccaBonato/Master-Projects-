import matplotlib.pyplot as plt
import numpy as np

def plot_simple(model_history):
    
    fig, axes = plt.subplots(1,4, figsize=(16,4))
    
    axes[0].plot(model_history.history["loss"], label="loss")
    axes[0].plot(model_history.history["val_loss"], label="val_loss")
    axes[0].plot( np.argmin(model_history.history["val_loss"]),
           np.min(model_history.history["val_loss"]),
            marker="x", color="r", label="best model")
    axes[0].legend();
    plt.xlabel("Epochs")

    axes[1].plot(model_history.history["dice_coef"], label="dice_coefficient")
    axes[1].plot(model_history.history["val_dice_coef"], label="val_dice_coefficient")
    axes[1].legend();
    plt.xlabel("Epochs")
    
    axes[2].plot(model_history.history["precision"], label="precision")
    axes[2].plot(model_history.history["val_precision"], label="val_precision")
    axes[2].legend();
    plt.xlabel("Epochs")

    axes[3].plot(model_history.history["recall"], label="recall")
    axes[3].plot(model_history.history["val_recall"], label="val_recall")
    axes[3].legend();
    plt.xlabel("Epochs")

    
def plot_Kfold(loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall):
    
    fig, axes = plt.subplots(len(loss),4, figsize=(24,5*len(loss)))
    
    for fold in np.arange(len(loss)):
        axes[fold,0].plot(loss[fold], label="loss")
        axes[fold,0].plot(val_loss[fold], label="val_loss")
        axes[fold,0].plot( np.argmin(val_loss[fold]),
               np.min(val_loss[fold]),
                marker="x", color="r", label="best model")
        axes[fold,0].legend();
        plt.xlabel("Epochs")

        axes[fold,1].plot(dice_coef[fold], label="dice_coefficient")
        axes[fold,1].plot(val_dice_coef[fold], label="val_dice_coefficient")
        axes[fold,1].legend();
        plt.xlabel("Epochs")

        axes[fold,2].plot(precision[fold], label="precision")
        axes[fold,2].plot(val_precision[fold], label="val_precision")
        axes[fold,2].legend();
        plt.xlabel("Epochs")

        axes[fold,3].plot(recall[fold], label="recall")
        axes[fold,3].plot(val_recall[fold], label="val_recall")
        axes[fold,3].legend();
        plt.xlabel("Epochs")
        
def plot_autocontext(step, num_folds, loss, val_loss, dice_coef, val_dice_coef, precision, val_precision, recall, val_recall):
    
    fig, axes = plt.subplots(step*num_folds,4, figsize=(16,4*num_folds*step))
    count=0
    for s in range(step):
        for f in range (num_folds):
            axes[s+f+count,0].plot(loss[f,s,:], label="loss")
            axes[s+f+count,0].plot(val_loss[f,s,:], label="val_loss")
            axes[s+f+count,0].plot( np.argmin(val_loss[f,s,:]),
                   np.min(val_loss[f,s,:]),
                    marker="x", color="r", label="best model")
            axes[s+f+count,0].legend();
            plt.xlabel("Epochs")

            axes[s+f+count,1].plot(dice_coef[f,s,:], label="dice_coefficient")
            axes[s+f+count,1].plot(val_dice_coef[f,s,:], label="val_dice_coefficient")
            axes[s+f+count,1].legend();
            plt.xlabel("Epochs")

            axes[s+f+count,2].plot(precision[f,s,:], label="precision")
            axes[s+f+count,2].plot(val_precision[f,s,:], label="val_precision")
            axes[s+f+count,2].legend();
            plt.xlabel("Epochs")

            axes[s+f+count,3].plot(recall[f,s,:], label="recall")
            axes[s+f+count,3].plot(val_recall[f,s,:], label="val_recall")
            axes[s+f+count,3].legend();
            plt.xlabel("Epochs")
            
        count+=num_folds-1