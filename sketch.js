function setup()
{
  var brain = new Nn(2, 2, 1);
  brain.setup();
  brain.train([0.23, 0.5], [1]);

}

function draw()
{

}
