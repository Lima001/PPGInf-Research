# This script defines a simple early stopping mechanism for deep learning model training.
# Early stopping is a technique used to prevent overfitting by halting the training process 
# once the model's performance starts to degrade. This is determined by  monitoring the 
# loss over epochs, and stopping when the loss does not improve  after a certain number
# of epochs (patience).

from numpy import inf

class EarlyStop():
    
    def __init__(self, patience=10, min_delta=0):
        """
            Initializes the EarlyStop object with the specified patience and minimum delta.
        
            Args:
                patience (int): Number of epochs with no improvement after which training will be stopped.
                min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
    
        self.patience = patience        # Number of epochs to wait before stopping
        self.min_delta = min_delta      # Minimum improvement threshold
        self.counter = 0                # Counter for tracking consecutive epochs without improvement
        self.min_validation_loss = inf  # Stores the minimum validation loss observed (initial value is positive infinity)
    
    def check(self, validation_loss):
        """
            Checks whether training should be stopped early based on the validation loss.
            
            Args:
                validation_loss (float): The current epoch's validation loss.
            
            Returns:
                bool: True if training should be stopped, False otherwise.
        """
        
        # If the current validation loss is an improvement, reset the counter and update the minimum loss.
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0

            return False    # Continue training
        
        # If no improvement, increment the counter
        self.counter += 1
        
        # Stop training if the counter exceeds the patience threshold
        return self.counter > self.patience