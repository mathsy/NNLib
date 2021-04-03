using System;
using System.Collections;
using System.Diagnostics;

namespace NNLib
{
    internal class NN
    {

        public class Basis
        {
            public abstract class Trainer
            {
                public abstract NeuralNetwork train(NeuralNetwork nn, Data.DataCollection data, Loss lossFunction, double maximumLoss);
                [Obsolete("Settings are still under heavy development")]

                public abstract void SetParameters(params Settings.TrainerSettings[] setting);

            }
            /// <summary>
            /// Base class of all Neurons
            /// </summary>
            public abstract class Neuron
            {
                public abstract double activation(double value);

                public double value = 0;


            }

            [Obsolete("Settings are still under heavy development")]
            public abstract class Setting
            {
                internal string tag = "";
                internal object value = "";

                public Setting()
                {
                    setTag("");
                }
                public Setting(object setting)
                {
                    value = setting;
                    setTag(setting);
                }
                /// <summary>
                /// Defines what the tag is
                /// </summary>
                /// <param name="setting"></param>
                public abstract void setTag(object setting);
               

            }

            /// <summary>
            /// Base class of all Losses and Loss-Calculators
            /// </summary>
            public abstract class Loss
            {
                public abstract double LossFunction(double error);

                public double CalculateLoss(NeuralNetwork nn, ref Data.DataCollection data)
                {
                    Data.Dataset[] datasets = data.getData();
                    double error = 0;
                    foreach (Data.Dataset dataset in datasets)
                    {
                        Basis.Neuron[] output = nn.Calculate(dataset.input);
                        int i = 0;
                        foreach (Basis.Neuron neuron in output)
                        {
                            try
                            {
                                double local_error = 0;
                                local_error += neuron.value - dataset.output[i];
                                error += LossFunction(local_error);
                            }
                            catch
                            { }


                        }

                    }
                    return error;
                }

            }



        }

        public class Data
        {

            public class Dataset
            {
                public int dimension = 0;
                public double[] input;
                public double[] output;


                public Dataset(double[] input_data, double[] outcome)
                {
                    dimension = input_data.Length;
                    input = input_data;
                    output = outcome;
                }
                public Dataset(int InputDataDimension, double[] input_data, double[] outcome)
                {
                    dimension = InputDataDimension;
                    input = input_data;
                    output = outcome;
                }
            }

            public class DataCollection
            {
                private ArrayList datasets = new ArrayList();
                /// <summary>
                /// Initiates a new DataCollection
                /// </summary>
                public DataCollection()
                {

                }
                /// <summary>
                /// Initiates a new DataCollection with initial data.
                /// </summary>
                /// <param name="data">The various Datasets</param>
                public DataCollection(params Dataset[] data)
                {
                    int dim = data[0].dimension;
                    foreach (Dataset d in data)
                    {
                        datasets.Add(d);
                        if (dim != d.dimension)
                        {
                            throw new NNExceptions.IncontinousDataDimensionException("The input data has varios dimensional datasets. They must all have the same dimension.");
                        }
                    }
                }

                /// <summary>
                /// Converts the Data saved in the DataCollection into an Array
                /// </summary>
                /// <returns>Returns an Array of Datasets</returns>
                public Dataset[] getData()
                {
                    Dataset[] data = new Dataset[datasets.Count];
                    int i = 0;
                    foreach (Dataset da in datasets)
                    {
                        data[i] = da;
                        i++;
                    }
                    return data;
                }
                /// <summary>
                /// Gets a random number of various datasets
                /// </summary>
                /// <param name="amount">Number of Datasets to choose</param>
                /// <returns>Random Dataset-Array</returns>
                public Dataset[] getData(int amount)
                {
                    Dataset[] data = new Dataset[amount];
                    Dataset[] complete = getData();
                    ArrayList indexer = new ArrayList();
                    Random random = new Random();
                    int i = 0;
                    while (indexer.Count <= amount)
                    {
                        int rand = random.Next(complete.Length + 1);
                        if (indexer.Contains(rand) == false)
                        {
                            indexer.Add(rand);
                            data[i] = complete[rand];
                            i++;
                        }

                    }

                    return data;
                }


            }

        }

        public class Neurons
        {
            /// <summary>
            /// Returns the value without any changes.
            /// </summary>
            public class OneByOneNeuron : Basis.Neuron
            {
                public override double activation(double value)
                {
                    return value;
                }
            }
            /// <summary>
            /// Squares the input signals
            /// </summary>
            public class SquaredNeuron : Basis.Neuron
            {
                public override double activation(double value)
                {
                    return Math.Pow(value, 2 / 1);
                }
            }
            /// <summary>
            /// Cubes the input signals
            /// </summary>
            public class CubedNeuron : Basis.Neuron
            {
                public override double activation(double value)
                {
                    return Math.Pow(value, 3 / 1);
                }
            }

            /// <summary>
            /// Returns the value when it is positive, else it returns 0.
            /// </summary>
            public class ReLUNeuron : Basis.Neuron
            {
                public override double activation(double value)
                {
                    return Math.Max(0, value);
                }
            }

            public class SqrtNeuron : Basis.Neuron
            {
                public override double activation(double value)
                {
                    return Math.Sqrt(value);
                }
            }

        }

        public class Trainer
        {

            public class GeneticTrainer : Basis.Trainer
            {
                private int population = 200;
                private int maximumItter = int.MaxValue;
                private NeuralNetwork initial;
                private NeuralNetwork best;
                private Basis.Loss loss;
                private Data.DataCollection datacollection;
                private double mutationrate = 0.01;
                private double maximumloss = 0;

                public override NeuralNetwork train(NeuralNetwork nn, Data.DataCollection data, Basis.Loss lossFunction, double maximumLoss)
                {
                    initial = nn;
                    best = nn;
                    maximumloss = maximumLoss;
                    loss = lossFunction;
                    datacollection = data;
                    return start();
                }

                [Obsolete("Settings are still under heavy development.")]
                public override void SetParameters(params Settings.TrainerSettings[] setting)
                {
                    foreach(Settings.TrainerSettings set in setting)
                    {
                        Type tp = set.GetType();
                        if(tp.Equals(typeof(Settings.MaximumItterationSetting)))
                        {
                            maximumItter = (int)set.value;
                        }
                        if(tp.Equals(typeof(Settings.MaximumLossSetting)))
                        {
                            switch((string)set.tag)
                            {
                                case ("low"):
                                    maximumloss = initial.weights.Length * 0.5;
                                    break;
                                case ("normal"):
                                    maximumloss = initial.weights.Length;
                                    break;
                                case ("high"):
                                    maximumloss = initial.weights.Length * 2;
                                    break;
                                default:
                                    maximumloss = Convert.ToInt32(set.tag);
                                break;
                            }
                        }
                    }
                }
                internal NeuralNetwork start()
                {
                    int itteration = 0;
                    double[] bestWeights = new double[initial.getWeights().Length];
                    double best_loss = double.MaxValue;
                    while (itteration < maximumItter)
                    {

                        Debug.WriteLine("Started itteration " + itteration);
                        NeuralNetwork[] pop = new NeuralNetwork[population];
                        Debug.WriteLine("Length of population array: " + pop.Length);
                        pop[0] = best;
                        int i = 1;
                        double[] localmutate = best.getWeights();
                        while (i < population)
                        {
                            NeuralNetwork local = new NeuralNetwork();
                            Debug.WriteLine("Info about local generated population network: " + local.getWeights().Length + " weights length");
                            local = best;

                            local = mutate(local);

                      
                            Debug.WriteLine("Make weights compare");
                            if (local.weights == localmutate)
                            {
                                local = mutate(local);
                                Debug.WriteLine("Equal weights");
                                //    goto Comparer;
                            }

                            pop[i] = local;
                            i++;
                        }



                        foreach (NeuralNetwork neuralNetwork in pop)
                        {
                            double current_loss = loss.CalculateLoss(neuralNetwork, ref datacollection);
                            Debug.WriteLine("Calculating loss");
                            if (current_loss < best_loss)
                            {
                                best_loss = current_loss;
                                best = neuralNetwork;
                                Debug.WriteLine("Current best loss:" + best_loss);
                            }

                        }
                        Console.WriteLine("Itteration " + itteration + ", global best loss: " + best_loss);
                        if (best_loss < maximumloss)
                        {
                            break;
                        }

                        itteration++;

                    }

                    return best;
                }

                private double randomizer()
                {
                    Random random = new Random();
                    return random.NextDouble();
                }
                private NeuralNetwork mutate(NeuralNetwork nn)
                {
                    string weight = "";
                    foreach (double d in nn.weights)
                    {
                        if (weight.Length > 30)
                        {
                            break;
                        }
                        weight += d + " ";
                    }
                    Debug.WriteLine(weight);
                    Random random = new Random();
                    int i = 0;
                    foreach (double d in nn.weights)
                    {
                        //  Debug.WriteLine("Before change: " + weights[i]);
                        double change = Math.Min(2 * (nn.weights[i] * mutationrate) * (random.NextDouble() - 0.5), 1);
                        nn.weights[i] += change;
                        //   Debug.WriteLine("After change: " + weights[i]);
                        //   Debug.WriteLine("Change of weights: " + change);
                        //   Debug.WriteLineIf(random.NextDouble() > 0.9999, "Current weight example: " + weights[i]);
                        i++;
                    }
                    return nn;
                }

            }

        }

        public class Loss
        {
            /// <summary>
            /// Returns the sum of all input errors without any further calculation
            /// </summary>
            public class AbsoluteLoss : Basis.Loss
            {
                public override double LossFunction(double error)
                {
                    return error;
                }
            }
            /// <summary>
            /// Returns the sum of all squared input errors.
            /// </summary>
            public class SquaredLoss : Basis.Loss
            {
                public override double LossFunction(double error)
                {
                    return Math.Pow(error, 2 / 1);
                }
            }

        }

        public class Layer
        {
            private static ArrayList neurons = new ArrayList();




            /// <summary>
            /// Initiates a new Neural Layer
            /// </summary>
            /// <param name="NeuronType"> The Neuron Type, found in the Neurons-Class</param>
            /// <param name="number"> The amount of neurons in this layer</param>
            /// 
            public Layer(Basis.Neuron NeuronType, int number)
            {
                number--;
                int i = 0;
                while (i < number)
                {
                    Basis.Neuron neuron = NeuronType;
                    neurons.Add(neuron);
                    i++;
                }

                Debug.WriteLine("Initialized Layer with " + neurons.Count + " neurons.");

            }
            /// <summary>
            /// Starts the calculation of this layer
            /// </summary>
            /// <param name="previousLayer">The result of the previous layer</param>
            /// <param name="weights">The Weights of the neural network</param>
            /// <param name="weightsIndex">The current "weights cursor"</param>
            /// <param name="newIndex">The new weights Index</param>
            /// <returns></returns>
            public Basis.Neuron[] calculate_layer(Basis.Neuron[] previousLayer, double[] weights, int weightsIndex, out int newIndex)
            {
                int i = 0;

                foreach (Basis.Neuron neuron in neurons)
                {
                    double current_value = 0;
                    foreach (Basis.Neuron neuron1 in previousLayer)
                    {
                        //Has to be fixed a bit more beautiful
                        if (weightsIndex == weights.Length)
                        {
                            weightsIndex--;
                        }
                        current_value += neuron1.value * weights[weightsIndex];
                        weightsIndex++;
                    }
                    Basis.Neuron n = (Basis.Neuron)neurons[i];
                    n.value = neuron.activation(current_value);
                    i++;
                }

                newIndex = weightsIndex;

                foreach (Basis.Neuron neuron2 in ConvertNeuronstoArray())
                {
                    Debug.WriteLine("Neuron Value: " + neuron2.value);
                }

                return ConvertNeuronstoArray();

            }

            private Basis.Neuron[] ConvertNeuronstoArray()
            {
                Basis.Neuron[] neuronarray = new Basis.Neuron[neurons.Count];

                int i = 0;
                foreach (Basis.Neuron neuron in neurons)
                {
                    neuronarray[i] = neuron;
                    i++;
                }
                return neuronarray;
            }

            public int numberOfNeuronsInLayer()
            {
                return neurons.Count;
            }

            public Basis.Neuron[] Input(params double[] InputValues)
            {
                int i = 0;
                try
                {
                    foreach (double value in InputValues)
                    {
                        Basis.Neuron n = (Basis.Neuron)neurons[i];
                        n.value = n.activation(value);
                    }
                }
                catch (IndexOutOfRangeException)
                {
                    throw new NNExceptions.TooLargeInputDimensionException("Found more input values than neurons in layer!");
                }

                return ConvertNeuronstoArray();
            }


        }

        public class NeuralNetwork
        {
            private static ArrayList layers = new ArrayList();
            public double[] weights = new double[0];
            private static int weights_index = 0;

            /// <summary>
            /// Initiates a new Neural Network
            /// </summary>
            /// <param name="layer">The layers in the neural network in the correct row </param>
            /// 
            public NeuralNetwork(params Layer[] layer)
            {
                foreach (Layer layer1 in layer)
                {
                    layers.Add(layer1);
                }

                setNumberOfWeights();
                randomweights();
            }
            /// <summary>
            /// Initiates a new Neural Network
            /// </summary>
            public NeuralNetwork()
            {
                setNumberOfWeights();
                randomweights();
            }

            public double[] getWeights()
            {
                return weights;
            }
            internal void setNumberOfWeights()
            {
                int number_weights = 0;

                int previous_layer = 0;
                foreach (Layer layer in layers)
                {
                    if (number_weights == 0)
                    {
                        Debug.WriteLine("Number of neurons in layer: " + layer.numberOfNeuronsInLayer());
                    }

                    number_weights += layer.numberOfNeuronsInLayer() * previous_layer;
                    previous_layer = layer.numberOfNeuronsInLayer();

                }

                weights = new double[number_weights];
            }
            /// <summary>
            /// Adds a layer at the end of the neural network!
            /// </summary>
            /// <param name="layer">The layer to add</param>
            public void AddLayer(Layer layer)
            {
                layers.Add(layer);
                setNumberOfWeights();
            }

            private void randomweights()
            {
                int i = 0;
                Random random = new Random();
                foreach (double d in weights)
                {
                    weights[i] = random.NextDouble();
                    i++;
                }

            }

            /// <summary>
            /// This method replaces the old weights
            /// </summary>
            /// <param name="newWeights">The new weights in a double array</param>
            public void set_weights(double[] newWeights)
            {
                if (newWeights.Length == weights.Length)
                {
                    weights = newWeights;

                }
                else
                {
                    throw new NNExceptions.WrongWeightsFormatException("The weights array to add has not the right length! Expected: " + weights.Length + ", but it has: " + newWeights.Length);
                }
            }

            /// <summary>
            /// Starts the calculation of the neural network
            /// </summary>
            /// <param name="input">Array of inputs</param>
            /// <returns>Array of calculated output neurons</returns>
            public Basis.Neuron[] Calculate(double[] input)
            {
                weights_index = 0;
                int layerindex = 0;
                Layer previousLayer = new Layer(new Neurons.OneByOneNeuron(), 1);
                Basis.Neuron[] result = new Basis.Neuron[0];
                foreach (Layer layer in layers)
                {
                    //Is it the Input layer currently calculated?
                    if (layerindex == 0)
                    {
                        result = layer.Input(input);
                        layerindex++;
                    }
                    else
                    {
                        result = layer.calculate_layer(result, weights, weights_index, out weights_index);
                        layerindex++;
                        Debug.WriteLine(weights_index + " " + layerindex);
                    }
                }

                return result;
            }




        }

        public class Settings
        {
            [Obsolete("Settings are still under heavy development")]

            internal class TrainerSettings : Basis.Setting
            {
               public TrainerSettings() { }
                public TrainerSettings(object setting)
                {
                    value = setting;
                    setTag(setting);
                }
                public override void setTag(object setting)
                {
                    
                }
            }
            [Obsolete("Settings are still under heavy development")]

            public class MaximumItterationSetting : TrainerSettings
            {
                public override void setTag(object setting)
                {
                    tag = "maxitter";
                }
            }
            [Obsolete("Settings are still under heavy development")]

            public class MaximumLossSetting : TrainerSettings
            {
                public MaximumLossSetting(string setting)
                {
                    value = setting;
                    setTag(setting);
                }
                public override void setTag(object setting)
                {
                    try
                    {
                        double valueparsed = 0;
                        if ((string)setting == "low")
                        {
                            tag = "low";
                        }
                        else if ((string)setting == "normal")
                        {
                            tag = "normal";
                        }
                        else if ((string)setting == "high")
                        {
                            tag = "high";
                        }

                        else if (double.TryParse((string)setting, out valueparsed))
                        {
                            tag = (string)setting;
                        }

                        else
                        {
                            throw new NNExceptions.WrongWeightsFormatException("The setting " + setting.ToString() + " couldn't be parsed");
                        }
                    }
                    catch
                    {
                        throw new NNExceptions.WrongWeightsFormatException("The setting " + setting.ToString() + " couldn't be parsed");

                    }
                }
            }




        }



        /// <summary>
        /// This class has all of the custom exceptions in the NNLib class library
        /// </summary>
        private class NNExceptions
        {

            /// <summary>
            /// The NetworkNotImprovingException is used when a trainer couldn't improve the network at all.
            /// </summary>
            [Serializable]
            public class NetworkNotImprovingException : Exception
            {
                public NetworkNotImprovingException() { }
                public NetworkNotImprovingException(string message) : base(message) { }
                public NetworkNotImprovingException(string message, Exception inner) : base(message, inner) { }
                protected NetworkNotImprovingException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
            }

            /// <summary>
            /// The WrongWeightsFormatException is used when weights are going to be replaced or read from string.
            /// </summary>
            [System.Serializable]
            public class WrongWeightsFormatException : Exception
            {
                public WrongWeightsFormatException() { }
                public WrongWeightsFormatException(string message) : base(message) { }
                public WrongWeightsFormatException(string message, Exception inner) : base(message, inner) { }
                protected WrongWeightsFormatException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
            }

            /// <summary>
            /// The IncontinousDataDimensionException is used when datasets in the same DataCollection have various dimensions.
            /// </summary>
            [Serializable]
            public class IncontinousDataDimensionException : Exception
            {
                public IncontinousDataDimensionException() { }
                public IncontinousDataDimensionException(string message) : base(message) { }
                public IncontinousDataDimensionException(string message, Exception inner) : base(message, inner) { }
                protected IncontinousDataDimensionException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
            }

            /// <summary>
            /// This Exception is thrown when there is not enough data for a specific routine
            /// </summary>
            [Serializable]
            public class NotEnoughDataException : Exception
            {
                public NotEnoughDataException() { }
                public NotEnoughDataException(string message) : base(message) { }
                public NotEnoughDataException(string message, Exception inner) : base(message, inner) { }
                protected NotEnoughDataException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
            }

            /// <summary>
            /// The TooLargeInputDimensionException is used when the input is higher-dimensional than the input layer.
            /// </summary>
            [Serializable]
            public class TooLargeInputDimensionException : Exception
            {
                public TooLargeInputDimensionException() { }
                public TooLargeInputDimensionException(string message) : base(message) { }
                public TooLargeInputDimensionException(string message, Exception inner) : base(message, inner) { }
                protected TooLargeInputDimensionException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
            }

            /// <summary>
            /// The WrongSettingTypeException is used when a settings is from a wrong type
            /// </summary>
            [Serializable]
            public class WrongSettingTypeException : Exception
            {
                public WrongSettingTypeException() { }
                public WrongSettingTypeException(string message) : base(message) { }
                public WrongSettingTypeException(string message, Exception inner) : base(message, inner) { }
                protected WrongSettingTypeException(
                  System.Runtime.Serialization.SerializationInfo info,
                  System.Runtime.Serialization.StreamingContext context) : base(info, context) { }
            }

        }

    }
}
