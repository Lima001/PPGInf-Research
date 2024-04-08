#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import inf


class EarlyStop():
    
    def __init__(self, patience=10, min_delta=0):
    
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = inf
    
    def check(self, validation_loss):
        
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0

            return False
            
        
        self.counter += 1
        return self.counter > self.patience
