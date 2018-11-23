function Nn(inputno, hiddenno, outputno)
{
  this.input_nodes = inputno;
  this.hidden_nodes = hiddenno;
  this.output_nodes = outputno;
  this.learning_rate = 0.01;
  this.weights_ih;
  this.weights_ho;
  this.bias_h;
  this.bias_o;
  this.weight_prev_ih;
  this.weight_prev_ho;
  this.bias_prev_h;
  this.bias_prev_o;
  this.momentum = 0.5;

  this.setup = function(learning_rate, momentum)
  {
    //Initializing the weights
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ih.setup();
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ho.setup();
    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_h.setup();
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_o.setup();

    //Setting the learning rate and momentum rate
    this.learning_rate = learning_rate;
    this.momentum      = momentum;

    //Creating storage for momentum variables
    this.weight_prev_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weight_prev_ih.setup();
    this.weight_prev_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weight_prev_ho.setup();
    this.bias_prev_h = new Matrix(this.hidden_nodes, 1);
    this.bias_prev_h.setup();
    this.bias_prev_o = new Matrix(this.output_nodes, 1);
    this.bias_prev_o.setup();
  }

  this.feedforward = function(inputs)
  {
    //Creating the output matrix
    var output = new Matrix(inputs.length, 1);

    //Converting the input into a matrix
    inputs = Matrix.tomatrix(inputs);

    //Generating the hidden layer output
    output = Matrix.multiply(this.weights_ih, inputs);
    output = Matrix.add(output, this.bias_h);

    //Passing output through activation function
    output.activate();

    //Generating the final output
    output = Matrix.multiply(this.weights_ho, output);
    output = Matrix.add(output, this.bias_o);

    //Passing output through activation function
    output.activate();

    //Changing the output into an array
    output = output.toarray();

    //Returning the output
    return output;
  }

  this.train = function(inputs, targets)
  {
    //Creating the output errors
    var error_o = new Matrix(targets.length, 1);

    var outputs = this.feedforward(inputs);

    //Converting the output into a matrix
    outputs = Matrix.tomatrix(outputs);

    //Converting the inputs to a matrix
    inputs = Matrix.tomatrix(inputs);

    //Changing the target to a martix
    targets = Matrix.tomatrix(targets);

    //Computing the output error
    error_o = Matrix.subtract(targets, outputs);

    //Normailzing the weights in the hidden layer
    var n_weights_ho = new Matrix(this.weights_ho.rows, this.weights_ho.cols);
    n_weights_ho.setup();

    //Assigning the normalized weights matrix
    for(var i = 0; i < n_weights_ho.rows; ++i)
    {
      var rowsum = this.weights_ho.sumofrow(i);
      for(var j = 0; j < n_weights_ho.cols; ++j)
      {
        n_weights_ho.matrix[i][j] = this.weights_ho.matrix[i][j] / rowsum;
      }
    }

    //Transposing this normalised weights matrix
    var n_weights_ho_t = Matrix.transpose(n_weights_ho);

    //Finding the error in the hidden layer
    var error_h = Matrix.multiply(n_weights_ho_t, error_o);

    //Computing the change in weights and bias in the hidden layer and assigning them
    var hidden_inputs = Matrix.multiply(this.weights_ih, inputs);
    hidden_inputs = Matrix.add(hidden_inputs, this.bias_h);
    hidden_inputs.activate();

    //Finding change in weights in output, hidden layer
    outputs.activateder();
    var gradient_ho = Matrix.hadmardproduct(outputs, error_o);
    var delta_w_ho = Matrix.multiply(gradient_ho, Matrix.transpose(hidden_inputs));
    
    //Multiplying the change by learning rate
    delta_w_ho.multiplyscaler(this.learning_rate);
    gradient_ho.multiplyscaler(this.learning_rate);

    //Computing momentum change
    this.weight_prev_ho.multiplyscaler(this.momentum);
    this.bias_prev_o.multiplyscaler(this.momentum); 
    //console.log(this.weight_prev_ho, this.bias_prev_o);    
    
    //Applying gradient descent
    this.weights_ho = Matrix.add(this.weights_ho, delta_w_ho);
    this.bias_o     = Matrix.add(this.bias_o, gradient_ho);

    //Applying momentum
    this.weights_ho = Matrix.add(this.weights_ho, this.weight_prev_ho);
    this.bias_o     = Matrix.add(this.bias_o, this.bias_prev_o);

    //Updating the previous weights
    this.weight_prev_ho = delta_w_ho;
    this.bias_prev_o    = gradient_ho;
    
    //Computing the change in weights of the input layer
    hidden_inputs.activateder();
    
    //Finding gradients in input hidden layer
    var gradient_ih = Matrix.hadmardproduct(hidden_inputs, error_h);
    var delta_w_ih  = Matrix.multiply(gradient_ih, Matrix.transpose(inputs));

    //Computing momentum change in input hidden layer
    this.weight_prev_ih.multiplyscaler(this.momentum);
    this.bias_prev_h.multiplyscaler(this.momentum);
    //console.log(this.weight_prev_ih, this.bias_prev_h);

    //Multiplying by learning rate
    delta_w_ih.multiplyscaler(this.learning_rate);
    gradient_ih.multiplyscaler(this.learning_rate);
    
    //Applying gradient descent    
    this.weights_ih = Matrix.add(this.weights_ih, delta_w_ih);
    this.bias_h     = Matrix.add(this.bias_h, gradient_ih);
    
    //Applying momentum
    this.weights_h = Matrix.add(this.weights_ih, this.weight_prev_ih);
    this.bias_h    = Matrix.add(this.bias_h, this.bias_prev_h);

    //Updating the previous weights
    this.weight_prev_ih = delta_w_ih;
    this.bias_prev_h    = gradient_ih;
    //console.log(this.weight_prev_ih);
  }

  //Function to predict given neural network
  this.predict = function(inputs)
  {
    return (this.feedforward(inputs));
  }
}
