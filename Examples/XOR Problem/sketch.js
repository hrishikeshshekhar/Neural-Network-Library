let training_data;
let nn;
let learning_rate = 0.01;
let size = 800;
let resol = 20;

function setup()
{
  createCanvas(800, 800);
  nn = new Nn(2, 20, 1);
  nn.setup(learning_rate);

  training_data = [
    {
      input: [0, 0],
      output: [0]
    },
    {
      input: [0, 1],
      output: [1]
    },
    {
      input: [1, 0],
      output: [1]
    },
    {
      input: [1, 1],
      output: [0]
    }
  ];
}

function draw()
{
  background(0);
  for(var i = 0; i < 1000; ++i)
  {
    var data = random(training_data);
    nn.train(data.input, data.output);
  }

  let cols = width / resol;
  let rows = width / resol;
  for(let x = 0; x < rows; ++x)
  {
    for(let y = 0; y < cols; ++y)
    {
      let x1 = x / rows;
      let x2 = y / cols;
      let inputs = [x1, x2];
      let r = nn.predict(inputs);
      fill(r * 255);
      rect(x * resol, y * resol, resol, resol);
    }
  }
}
