
import random
import math
import operator as op
import copy

# parameters of cartesian genetic programming
MUT_PB = 0.015  # mutate probability
N_COLS = 500   # number of cols (nodes) in a single-row CGP
LEVEL_BACK = 500  # how many levels back are allowed for inputs in CGP

# parameters of evolutionary strategy: MU+LAMBDA
MU = 2
LAMBDA = 8
N_GEN = 4  # max number of generations

# if True, then additional information will be printed
VERBOSE = False


class Function:
    """
    A general function
    """
    def __init__(self, f, arity):
        self.f = f
        self.arity = arity

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

class Node:
    """
    A node in CGP graph
    """
    def __init__(self, max_arity):
        """
        Initialize this node randomly
        """
        self.i_func = None
        self.i_inputs = [None] * max_arity
        self.weights = [None] * max_arity
        self.i_output = None
        self.output = None
        self.active = False

class Individual:
    """
    An individual (chromosome, genotype, etc.) in evolution
    """
    function_set = None
    weight_range = [-1, 1]
    max_arity = 3
    n_inputs = 8
    n_outputs = 4
    n_cols = N_COLS
    level_back = LEVEL_BACK

    def __init__(self):
        self.nodes = []
        for pos in range(self.n_cols):
            self.nodes.append(self._create_random_node(pos))
        for i in range(1, self.n_outputs + 1):
            self.nodes[-i].active = True
        self.fitness = None
        self._active_determined = False

    def _create_random_node(self, pos):
        node = Node(self.max_arity)
        node.i_func = random.randint(0, len(self.function_set) - 1)
        for i in range(self.function_set[node.i_func].arity):
            node.i_inputs[i] = random.randint(max(pos - self.level_back, -self.n_inputs), pos - 1)
            node.weights[i] = random.uniform(self.weight_range[0], self.weight_range[1])
        node.i_output = pos

        return node

    def _determine_active_nodes(self):
        """
        Determine which nodes in the CGP graph are active
        """
        # check each node in reverse order
        n_active = 0
        for node in reversed(self.nodes):
            if node.active:
                n_active += 1
                for i in range(self.function_set[node.i_func].arity):
                    i_input = node.i_inputs[i]
                    if i_input >= 0:  # a node (not an input)
                        self.nodes[i_input].active = True
        if VERBOSE:
            print("# active genes: ", n_active)

    def eval(self, *args):
        """
        Given inputs, evaluate the output of this CGP individual.
        :return the final output value
        """
        if not self._active_determined:
            self._determine_active_nodes()
            self._active_determined = True
        # forward pass: evaluate
        for node in self.nodes:
            if node.active:
                inputs = []
                for i in range(self.function_set[node.i_func].arity):
                    i_input = node.i_inputs[i]
                    w = node.weights[i]
                    if i_input < 0:
                        inputs.append(args[-i_input - 1] * w)
                    else:
                        inputs.append(self.nodes[i_input].output * w)
                node.output = self.function_set[node.i_func](*inputs)

        out, action = -float('Inf'), None
        for i in range(self.n_outputs):
        	if(self.nodes[-i].output > out):
        		action = i
        		out = self.nodes[-i].output
        
        return action

    def mutate(self, mut_rate=0.01):
        """
        Mutate this individual. Each gene is varied with probability *mut_rate*.
        :param mut_rate: mutation probability
        :return a child after mutation
        """
        child = copy.deepcopy(self)
        for pos, node in enumerate(child.nodes):
            # mutate the function gene
            if random.random() < mut_rate:
                node.i_func = random.choice(range(len(self.function_set)))
            # mutate the input genes (connection genes)
            arity = self.function_set[node.i_func].arity
            for i in range(arity):
                if node.i_inputs[i] is None or random.random() < mut_rate:  # if the mutated function requires more arguments, then the last ones are None 
                    node.i_inputs[i] = random.randint(max(pos - self.level_back, -self.n_inputs), pos - 1)
                if node.weights[i] is None or random.random() < mut_rate:
                    node.weights[i] = random.uniform(self.weight_range[0], self.weight_range[1])
            # initially an individual is not active except hte last output node
            node.active = False
        for i in range(1, self.n_outputs + 1):
            child.nodes[-i].active = True
        child.fitness = None
        child._active_determined = False
        return child



# function set
def protected_div(a, b):
    if abs(b) < 1e-6:
        return a
    return a / b


fs = [Function(op.add, 2), Function(op.sub, 2), Function(op.mul, 2), Function(protected_div, 2), Function(op.neg, 1)]
Individual.function_set = fs
Individual.max_arity = max(f.arity for f in fs)

def evolve(pop, fitness, mut_rate, mu, lambda_):
    """
    Evolve the population *pop* using the mu + lambda evolutionary strategy

    :param pop: a list of individuals, whose size is mu + lambda. The first mu ones are previous parents.
    :param mut_rate: mutation rate
    :return: a new generation of individuals of the same size
    """
    pop = [x for _,x in sorted(zip(fitness, pop))]
    parents = pop[-mu:]
    # generate lambda new children via mutation
    offspring = []
    for _ in range(lambda_):
        parent = random.choice(parents)
        offspring.append(parent.mutate(mut_rate))
    return parents + offspring


def create_population(n):
    """
    Create a random population composed of n individuals.
    """
    return [Individual() for _ in range(n)]
