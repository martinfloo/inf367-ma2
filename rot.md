
&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;


## ROT: 

### OLD TRAIN 1:

    ### MAIN FUNCTION FOR TRAINING ###
    def train(self, shots=1000, learning_rate=0.1, eps=0.1):
        # Run n_batches for each epoch
        loss_list = []
        
        for epoch in range(self.n_epochs):
            loss_list.append([])
            for batch in range(self.n_batches):
                # Initialize QNN with current "input_values" for encoding:
                self.QNN.create_circuit(X[batch])
                
                # Initialize start params 
                if (epoch == 0 and batch == 0):
                    self._initialize_params()
                else:
                    self._set_params(self.QNN.param_values, False)
                
                #print(f"------ BATCH {batch+1} ------")
                #print("Current Params:", self.QNN.param_values, "\n")
                #print("Current QC:\n\n", self.QNN.qc, "\n")
               

                # if (decode == binary and loss = CrossEntropy ): 
                actual, decoded_result = self._bin_decode(self._exec_circuit(shots=1000), batch, shots)
                loss = self.loss_func.compute_loss_CE(decoded_result, actual)
                loss_list[epoch].append(loss)
                gradients = self._finiteDifference(eps, batch)
                
                #print("Loss:", loss, "\n")
                
                self._update_param_vals(learning_rate=learning_rate, gradients=gradients)
                
                #print("Gradients:", gradients, "\n")
                #print("Params after grad_descent: ", self.QNN.param_values, "\n------ BATCH COMPL. ------\n\n\n\n")
            
            
        return loss_list
        
        
### OLD VIZ 1:

        ### VIZ LOSS ###
        def visualize_loss(loss_list):
            plt.figure(figsize=(12, 10))
            plt.plot(range(len(loss_list)), loss_list, label="Training Loss", color="blue")

            for i in range(4):
                plt.axvline(x=(i+1)*105, color='red', linestyle=':', linewidth=2, label=f"Epoch: {i+1}")
                plt.text((i+1)*115, max(loss_list), f'Epoch: {i+1}', color='red', ha='right', va='top')

            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.title("Training Loss per batch (singular batches)")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        visualize_loss(np.array(loss_list).flatten().tolist())
        
        
### OLD TESTER 1:
        
        
        # Very very very ugly training accuracy calculation
        trainers = [trainer1, trainer2, trainer3]

        for trainer in trainers:
            print("### TESTING TRAINED MODEL ###\n")
            print("Optimal Params:\n", myQNN.param_values, "\n")
            correct = 0
            num_possible_corr = len(y_test)
            backend = QasmSimulator()

            for X,y in zip(X_test, y_test):
                    myQNN.create_circuit(X)
                    myQNN.qc.assign_parameters({myQNN.params: trainer.best_params}, inplace=True)
                    transpiled_qc = transpile(myQNN.qc, backend)
                    job = backend.run(transpiled_qc, shots=1000)


                    counts = job.result().get_counts(myQNN.qc) 
                    probabilities = np.zeros(3)                    

                    for bitstring, count in counts.items():
                        classIndex = int(bitstring, 2) % 3  
                        probabilities[classIndex] += count

                    probabilities /= np.sum(probabilities)
                    prediction = np.argmax(probabilities)

                    if prediction == y:
                        correct +=1

            print(f"Accuracy on test data: {correct/num_possible_corr}")
            
            
### OLD TRAINER FUNCS:
    
    ### MAIN FUNCTION 1 FOR TRAINING ###
    def train_single_datapoint_upd(self, shots=1000, learning_rate=0.1, eps=0.1):
        print(f"#### INITIALIZING TRAINING (single datapoint updates) #####")
        start_total = t.time()
        
        # Run n_batches for each epoch
        loss_list_train = np.zeros(self.n_epochs)
        loss_list_val = np.zeros(self.n_epochs)
        
        for epoch in range(self.n_epochs):
            start = t.time()
            for batch in range(self.n_batches):
                # Initialize QNN with current "input_values" for encoding:
                self.QNN.create_circuit(X[batch])
                
                # Initialize start params 
                if (epoch == 0 and batch == 0):
                    self._initialize_params()
                else:
                    self._set_params(self.QNN.param_values, False)
                
                actual, decoded_result = self._bin_decode(self._exec_circuit(shots=shots), batch, shots)
                loss = self.loss_func.compute_loss_CE(decoded_result, actual)
                loss_list_train[epoch]+=loss
                gradients = self._finiteDifference(eps, batch)
                
                self._update_param_vals(learning_rate=learning_rate, gradients=gradients)
                
            # Update loss at the end of each epoch & Early stopping check
            loss_list_train[epoch] = loss_list_train[epoch]/len(self.X)            
            val_loss = self._val_loss(shots)
            loss_list_val[epoch] = val_loss
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_params = self.QNN.param_values.copy()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Check if patience is exceeded
            if self.no_improvement_count >= self.patience:
                end = t.time()
                print(f"Early stopping at Epoch {epoch + 1} (Time used: {(end-start)/60:.2f} min)")
                break
            
            end = t.time()
            print(f"Epoch {epoch+1} Finished (Time used: {(end-start)/60:.2f} min)")
        
        
        end_total = t.time()
        print(f"Training finished! (Time used: {(end_total-start_total)/60:.2f} min)")
        print(f"#### TRAINING ENDED (single datapoint updates) ##### \n\n\n")
            
        return loss_list_train, loss_list_val
    
    
    ### MAIN FUNCTION 2 FOR TRAINING ###
    def train_mini_batch_updates(self, shots=1000, learning_rate=0.1, eps=0.1,  batch_size=10, full_batch=False):
        if not full_batch:
            print(f"#### INITIALIZING TRAINING (mini batch updates) #####")
        else:
            print(f"#### INITIALIZING TRAINING (full batch updates) #####")
        start_total = t.time()
        # Run n_batches for each epoch
        loss_list_train = np.zeros(self.n_epochs)
        loss_list_val = np.zeros(self.n_epochs)
        
        for epoch in range(self.n_epochs):
            start = t.time()
            
            # Define batch size, shuffle and compute batches for each epoch
            indices_train = np.random.permutation(len(self.X))
            
            X_shuffled = self.X[indices_train]
            y_shuffled = self.y[indices_train]
            
            batches = [X_shuffled[i:i + batch_size] for i in range(0, len(X_shuffled), batch_size)]
            
            for batch_nr, batch in enumerate(batches):
                
                gradients = []
                for datapoint_idx, datapoint in enumerate(batch):
                    # Initialize QNN with current "input_values" for encoding:
                    self.QNN.create_circuit(datapoint)

                    # Initialize start params:
                    if (epoch == 0 and batch_nr == 0):
                        self._initialize_params()
                    else:
                        self._set_params(self.QNN.param_values, False)

                    actual, decoded_result = self._bin_decode(self._exec_circuit(shots=1000),
                                                              indices_train[(batch_nr*batch_size)+datapoint_idx],
                                                              shots)
                    
                    loss = self.loss_func.compute_loss_CE(decoded_result, actual)
                    loss_list_train[epoch] += loss
                    gradients.append(self._finiteDifference(eps, indices_train[(batch_nr*batch_size)+datapoint_idx]))
                
                
                avg_gradients = np.mean(gradients, axis=0)
                self._update_param_vals(learning_rate=learning_rate, gradients=avg_gradients)
                
                
            # Update loss at the end of each epoch & Early stopping check
            loss_list_train[epoch] = loss_list_train[epoch]/len(self.X)            
            val_loss = self._val_loss(shots)
            loss_list_val[epoch] = val_loss
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_params = self.QNN.param_values.copy()
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Check if patience is exceeded
            if self.no_improvement_count >= self.patience:
                end = t.time()
                print(f"Early stopping at Epoch {epoch + 1} (Time used: {(end-start)/60:.2f} min)")
                break
            
            end = t.time()
            print(f"Epoch {epoch+1} Finished (Time used: {(end-start)/60:.2f} min)")
            
        end_total = t.time()
        print(f"\nTraining finished! (Time used: {(end_total-start_total)/60:.2f} min)")
        if not full_batch:
            print(f"#### TRAINING ENDED (mini batch updates) ##### \n\n\n")
        else:
            print(f"#### TRAINING ENDED (full batch updates) ##### \n\n\n")
            
        return loss_list_train, loss_list_val
    
    
    ### MAIN FUNCTION 3 FOR TRAINING ###
    def train_full_batch_updates(self, shots=1000, learning_rate=0.1, eps=0.1):
        return self.train_mini_batch_updates(shots, learning_rate, eps, batch_size=len(self.X), full_batch=True)
    