#Paki dana - 30.01. leta gospodnjeg - 2022. pravi ovaj fajl kako bi resio probleme sa pamcenjem

#JEDNODIMENZIONA OPTIMIZACIJA

#FIBONACI
def fibonaci_metod(a, b, tol):
    #inicijalizacija n
    n = 1
    while((b - a) / tol) > fibonaci_broj(n):
        n += 1

    #inicijalizacija x1 i x2
    x1 = a + fibonaci_broj(n - 2) / fibonaci_broj(n) * (b - a)
    x2 = a + b - x1

    #petlja
    for i in range(2, n + 1):
        if func(x1) < func(x2):
            b = x2
            x2 = x1
            x1 = a + b - x2
        else:
            a = x1
            x1 = x2
            x2 = a + b - x1

    #zapis rezultata
    if func(x1) < func(x2):
        xopt = x1
        fopt = func(xopt)
    else:
        xopt = x2
        fopt = func(x2)

    return xopt, fopt, n

def fibonaci_broj(n):
    #1 1 2 3 5 8...
    if n < 3:
        f = 1
    else:
        fp = 1
        fpp = 1
        for i in range(3, n + 1):
            f = fp + fpp
            fpp = fp
            fp = f
    return f
#[]

#ZLATNI PRESEK
def zlatniPresek(a, b, tol):
    #inicijalizacija c
    c = (3 - math.sqrt(5)) / 2

    #inicijalizacija x1 i x2
    x1 = a + c * (b - a)
    x2 = a + b - x1
    n = 1

    #petlja
    while(b - a) > tol:
        n += 1
        if func(x1) <= func(x2):
            b = x2
            x1 = a + c * (b - a)
            x2 = a + b - x1
        else:
            a = x1
            x1 = a + c * (b - a)
            x2 = a + b - x1

    #zapis rezultata
    if func(x1) < func(x2):
        xopt = x1
        fopt = func(x1)
    else:
        xopt = x2
        fopt = func(x2)

    return xopt, fopt, n
#[]

#NJUTN-RAPSON
def newtonRaphnson(x0, tol):
    #inicijalizacija
    x_novo = x0
    x_pre = math.inf                                                                                                    #velika vrednost, kako bismo bili sigurni da ulazimo u while
    iteracije = 0

    #petlja
    while(abs(x_pre - x_novo) > tol):
        iteracije += 1
        x_pre = x_novo
        x_novo = x_pre - dfunc(x_pre) / ddfunc(x_pre)

    #zapis rezultata
    xopt = x_novo
    fopt = func(xopt)
    return xopt, fopt, iteracije
#[]

#SECICA
def secica(x1, x0, tol):
    #inicijalizacija
    x_ppre = math.inf
    x_pre = x0
    x_novo = x1
    iteracije = 0

    #petlja
    while(abs(x_novo - x_pre) > tol):
        iteracije += 1
        x_ppre = x_pre
        x_pre = x_novo
        x_novo = x_pre - dfunc(x_pre) *  (x_pre - x_ppre) / (dfunc(x_pre) - dfunc(x_ppre))

    #zapis rezultata
    xopt = x_novo
    fopt = func(xopt)
    return xopt, fopt, iteracije
#[]

#PARABOLA
def parabola(x1, x3, tol):
    X = np.array([x1, (x1 + x3)/2, x3]).transpose()
    pom = np.array([1, 1, 1]).transpose()
    Y = np.array([pom, X, X*X]).transpose()
    #x = [x1 x2 x3]'
    #pom = [1 1 1]'
    #Y = [1 1 1; x1 x2 x3; x1^2 x2^2 x3^2]
    F = np.linspace(0, 0, len(X))
    for i in range(0, len(X), 1):
        F[i] = func(X[i])

        abc = lin.solve(Y, F)                           #resava Ax = b ,matricnu jednacinu

        x = -abc[1] / 2 / abc[2]                        #abc[0] = A, abc[1] = B, abc[2] = C
        fx = func(x)
        n = 0

        while np.abs(np.dot([1, x, x**2], abc) - fx) > tol:
            if (x > X[1]) and (x < X[2]):
                if(fx < F[1]) and (fx < F[2]):
                    X = np.array([X[1], x, X[2]])
                    F = np.array([F[1], fx, F[2]])
                elif(fx > F[1]) and (fx < F[2]):
                    X = np.array([X[0], X[1], x])
                    F = np.array([F[0], F[1], fx])
                else:
                    print("Greska!")
            elif(x > X[0]) and (x < X[2]):
                if(fx < F[0]) and (fx < F[1]):
                    X = np.array([X[0], x, X[2]])
                    F = np.array([F[0], fx, F[2]])
                elif(fx > F[1]) and (fx < F[0]):
                    X = np.array([x, X[1], X[2]])
                    F = np.array([fx, F[1], F[2]])
                else:
                    print("Greska!")
            else:
                print("x lezi van granica!")

            pom = np.array([1, 1, 1]).transpose()
            Y = np.array([pom, X, X*X]).transpose()
            F = np.linspace(0, 0, len(X))
            for i in range(0, len(X), 1):
                F[i] = func(X[i])

            abc = lin.solve(Y, F)                           #resava Ax = b ,matricnu jednacinu

            x = -abc[1] / 2 / abc[2]                        #abc[0] = A, abc[1] = B, abc[2] = C
            fx = func(x)
            n = n + 1

            return x, fx, n
#[]



#GENETSKI ALGORITAM

#KODOVANJE HROMOZOMA - BINARNI
def bin_encode(chromosome, bin_val, min_val, precision):  # funkcija koja enkoduje jedan hromozom iz realnog u binarni broj u odnosu na parametre
    # chromosome - vrednost jedne koordinate, nor realna vrednost
    # bin_val - nivo diskretizacije
    # min_val - minimalna vrednost opsega
    # precision - broj bita za predstavljanje svakog broja

    ret = ""
    for g in chromosome:
        val = round((g - min_val) / bin_val)
        ret += bin(val)[2:].rjust(precision, '0')  # transformisemo u binarni broj
    return ret

def bin_encode_chromosomes(chromosomes, precision, max_val, min_val):  # enkodujemo celu listu hromozoma u binarne brojeve
    # chromosomes - ili jedinka
    # precision - broj bita za predstavljanje svakog broja
    # max_val, min_val - definišemo maksimalnu i minimalnu  vrednost opsega
    bin_val = (max_val - min_val) / (2 ** precision - 1)  # nivo diskretizicije
    # za svaki hromozom unutar jedinke (npr i za x i y vrednost) pretvaramo u binarni broj
    bin_chromosomes = [bin_encode(c, bin_val, min_val, precision) for c in chromosomes]
    return bin_chromosomes  # povratna vrednost su kodovani hromozomi
#[]

#DEKODOVANJE HROMOZOMA - BINARNI
def bin_decode(chromosome, bin_val, min_val, precision):  # funkcija koja dekoduje jedan hromozom iz binarnog u realni broj u odnosu na parametre

    ret = []
    for idx in range(0, len(chromosome), precision):
        g = int(chromosome[idx:idx + precision], 2)
        ret.append(g * bin_val + min_val)

    return ret

def bin_decode_chromosomes(chromosomes, precision, max_val, min_val):  # funkcija koja dekoduje sve hromozome iz liste istih

    bin_val = (max_val - min_val) / (2 ** precision - 1)

    bin_chromosomes = [bin_decode(c, bin_val, min_val, precision) for c in chromosomes]
    return bin_chromosomes
#[]

#GENERISANJE POCETNE POPULACIJE - OBA
#kreiraju se length broj jedinki za svaku dimenziju
def generate_inital_chromosomes(length, max, min, pop_size):
  return [ [random.uniform(min,max) for j in range(length)] for i in range(pop_size)]

#funkcija koja vraca prvi elementi liste costs, kao i prosecnu vrednost prilagodjenosti
def population_stats(costs):
  return costs[0], sum(costs)/len(costs)
#[]

#RANGIRANJE I SELEKCIJA JEDINKI - OBA
# sortiranje hromozoma na osnovu prilagodjenosti
# cost - funkcija koja odredjuje prilagodjenost
# chromosomes - lista hromozoma
def rank_chromosomes(cost, chromosomes):
    costs = list(map(cost, chromosomes))                                                                                #pravi se lista prilagodjenosti
    ranked = sorted(list(zip(chromosomes, costs)), key=lambda c: c[1])                                                  #pravi se uredjeni par sortirani roditelji - prilagodjenosti

    return list(zip(*ranked))                                                                                           #lista uredjenih parova je povratna vrednost

# povratna vrednost funkcije je lista hromozoma na indeksima od 0 do n_keep pocetne liste chromosomes
def natural_selection(chromosomes, n_keep):
    return chromosomes[:n_keep]

#ruletska selekcija roditelja
#prednost se daja onim roditeljima sa najmanjom prilagodjenoscu
#popunjava se lista pairs koja je i povratna vrednost funkcije
def roulette_selection(parents):
    pairs = []                                                                                                          #lista parova koja ce biti povratna vrednost funkcije
    i = 0
    for i in range(0, len(parents), 2):                                                                                 #ako imam 6 roditelja, bice 3 iteracije, u svakoj se odrede po dva roditelja sa najvecim kooeficijentom

        weights = [];                                                                                                   #privremena lista
        for i in range(len(parents)):                                                                                   #prolazim kroz sve roditelje
            weights.append( (len(parents) - i) * random.random())    #za minimum                                        #ubacujem u listu weights roditelje, pravilo je da se najgori roditelj mnozi najvecim faktorom kako bi mu se dala prednost
        #  weights.append((i+1)*random.random())                     #za maksimum                                       #u ovoj situaciji se najgori roditelj mnozi najmanjim faktorom
        if (weights[0] >= weights[1]):                                                                                  #porede se samo prva dva elementa u listi
            maxInd1 = 0;                                                                                                #postavljaju se vrednosti flagova maxInd1 i 2 koje predstavljaju elemente liste sa najvecom prilagodjenosti
            maxInd2 = 1;
        else:
            maxInd1 = 1;
            maxInd2 = 0;

        for i in range(2, len(parents)):                                                                                #prolazi se kroz sve ostale elemente
            if weights[i] > weights[maxInd1]:                                                                           #flagove se eventualno menjaju
                maxInd2 = maxInd1
                maxInd1 = i
            elif weights[i] > weights[maxInd2]:
                maxInd2 = 1
        pairs.append([parents[maxInd1], parents[maxInd2]])                                                              #u listu pairs se dodaju roditelji sa najvecom prilagodjenosti

    return pairs
#[]

#1-TACKASTO UKRSTANJE - BINARNI
def one_point_crossover(pairs):
    length = len(pairs[0])                                                                                              #duzina jedne jedinke
    children = []                                                                                                       #lista dece

    for (a, b) in pairs:                                                                                                #za svaka dva roditelja

        r = random.randrange(0, length)                                                                                 #random generisana tacka ukrstanja [0, duzina_roditelja - 1]

        children.append(a[:r] + b[r:])                                                                                  #generisanje dece
        children.append(b[:r] + a[r:])

    return children
#[]

#2-TACKASTO UKRSTANJE - BINARNI
def two_point_crossover(pairs):
    length = len(pairs[0])                                                                                              #duzina jedne jedinke
    children = []                                                                                                       #lista dece

    for (a, b) in pairs:                                                                                                #za svaka dva roditelja

        r1 = random.randrange(0, length)                                                                                #random generisana tacka ukrstanja [0, duzina_roditelja - 1]
        r2 = random.randrange(0, length)                                                                                #random generisana tacka ukrstanja [0, duzina_roditelja - 1]

        if r1 < r2:
            children.append(a[:r1] + b[r1:r2] + a[r2:])                                                                 #generisanje dece
            children.append(b[:r1] + a[r1:r2] + b[r2:])
        else:
            children.append(a[:r2] + b[r2:r1] + a[r1:])
            children.append(b[:r2] + a[r2:r1] + b[r1:])

    return children
#[]

#UKRSTANJE - REALNI
def crossover(pairs):
    children = []                                                                                                       #lista dece

    for a, b in pairs:                                                                                                  #za svaka dva elementa u listi pairs

        r = random.random()                                                                                             #random broj u opsegu [0, 1]
        y1 = []                                                                                                         #deca su uredjeni parovi koji opisuju koordinatu
        y2 = []
        for i in range(0, len(a)):
            y1.append(r * a[i] + (1 - r) * b[i])                                                                        #formiranje dece po formuli za svaki par roditelja
            y2.append((1 - r) * a[i] + r * b[i])
        children.append(y1)                                                                                             #dodavanje u listu dece
        children.append(y2)

    return children
#[]

#MUTACIJA INVERZIJOM - BINARNI
def inv_mutation(chromosomes, mutation_rate):
    mutated_chromosomes = []                                                                                            #lista mutiranih hromozoma

    for chromosome in chromosomes:                                                                                      #za svaki hromozom

        if random.random() < mutation_rate:                                                                             #ako je random broj manji od granice, za sta je mala verovatnoca
            r1 = random.randrange(0, len(chromosome) - 1)                                                               #generise se random broj [0, duzina_roditelja - 2]
            r2 = random.randrange(0, len(chromosome) - 1)

            if r1 < r2:                                                                                                 #vrsi se invertovanje bita izmedju dva random broja
                mutated_chromosomes.append(chromosome[:r1] + chromosome[r1:r2][::-1] + chromosome[r2:])
            else:
                mutated_chromosomes.append(chromosome[:r2] + chromosome[r2:r1][::-1] + chromosome[r1:])

        else:
            mutated_chromosomes.append(chromosome)                                                                      #ako random broj nije manji od granice, u vecini slucajeva, samo se prepisuje isti hromozom

    return mutated_chromosomes
#[]

#MUTACIJA ROTACIJOM - BINARNI
def mutation(chromosomes, mutation_rate):
    mutated_chromosomes = []                                                                                            #lista mutiranih hromozoma
    for chromosome in chromosomes:                                                                                      #za svaki hromozom

        if random.random() < mutation_rate:                                                                             #ako je random broj manji od granice, za sta je mala verovatnoca
            r1 = random.randrange(0, len(chromosome) - 1)                                                               #generise se random broj [0, duzina_roditelja - 2]
            mutated_chromosomes.append(chromosome[:r1] + str(1 - int(chromosome[r1])) + chromosome[r1 + 1:])            #vrsi se inverzija samo jednog bita
        else:
            mutated_chromosomes.append(chromosome)                                                                      #prepisuje se pocetni hromozom neizmenjen

    return mutated_chromosomes
#[]

#MUTACIJA - REALNI
def mutation(chromosomes, mutation_rate, mutation_width):
    mutated_chromosomes = []                                                                                            #lista hromozoma
    for chromosome in chromosomes:                                                                                      #za svaki hromozom
        y1 = []                                                                                                         #lista koja sadrzi elemente nakon mutacije
        for i in range(0, len(chromosome)):                                                                             #za svaku dimenziju
            if random.random() < mutation_rate:                                                                         #generise se slucajan broj i proverava da li je manji od zadate granice
                r = random.random()                                                                                     #ako jeste onda ga preuzimam u prom. r

                y1.append(chromosome[i] + mutation_width * 2 * (r - 0.5))                                               #zatim menjam pocetni hromozom pomocu ulaznog parametra mutation_width i random broja, ubacujem u listu y1
            else:
                y1.append(chromosome[i])                                                                                #ako broj nije odgovarajuci ne menjam pocetni hromozom, ubacujem u listu y1

        mutated_chromosomes.append(y1)                                                                                  #dodajem izmenjen ili ne element u listu mutiranih hromozoma
    return mutated_chromosomes
#[]

#ELITIZAM - OBA
def elitis(chromosomes_old, chromosomes_new, elitis_rate, population_size):
    old_ind_size = int(np.round(population_size * elitis_rate))                                                         #definise se broj jedinki koje ce biti sacuvane na osnovu velicine populacije i verovatnoce elitizma
    return chromosomes_old[:old_ind_size] + chromosomes_new[:(population_size - old_ind_size)]                          #povratna vrednost je lista hromozoma, gde su prvim x stare jedinke, a preostale su nove
#[]



#PSO ALGORITAM

#KLASA CESTICE
class Particle:
    def __init__(self, x0, num_dimensions, options):
        self.velocity_i = []                                                                                            #trenutna brzina cestice
        self.position_i = []                                                                                            #trenutna pozicija cestice
        self.pos_best_i = []                                                                                            #najbolja pozicija cestice
        self.fitness_i = -1                                                                                             #trenutna prilagodjenost cestice
        self.fitness_best_i = -1                                                                                        #najbolja prilagodjenost cestice
        self.num_dimensions = num_dimensions                                                                            #broj dimenzija

        # Inicijalne pozicije i brzine
        for i in range(0, num_dimensions):                                                                              #za svaku dimenziju
            self.velocity_i.append( (np.random.rand() - 0.5) * 2 * options.vspaninit )                                  #random generisana inicijalna brzina cestice
            self.position_i.append(x0[i][0])                                                                            #preuzeta pocetna pozicija iz liste x0

    # Izracunavanje prilagodjenosti cestice i update vrednosti
    def evaluate(self, costFunc):
        self.fitness_i = costFunc(self.position_i)                                                                      #izracunavanje prilagodjenosti na osnovu trenutnih pozicija
        # update individualnih rekorda
        if self.fitness_i < self.fitness_best_i or self.fitness_best_i == -1:                                           #ako je trenutna prilagodjenost manja/bolja od najbolje, update
            self.pos_best_i = self.position_i                                                                           #update-uje se najbolja pozicija
            self.fitness_best_i = self.fitness_i                                                                        #kao i najbolja prilagodjenost

    def linrate(self, xmax, xmin, tmax, tmin, t):                                                                       #???
        x = xmin + ((xmax - xmin) / (tmax - tmin)) * (tmax - t)
        return x

    # Izracunavanje nove brzine cestice
    def update_velocity(self, pos_best_g, maxiter, iter, opt):

        # Izracunavanje PSO parametara
        w = self.linrate(opt.wf, opt.wi, maxiter, 0, iter);
        cp = self.linrate(opt.cbf, opt.cbi, maxiter, 0, iter);
        cg = self.linrate(opt.cgf, opt.cgi, maxiter, 0, iter);

        for i in range(0, self.num_dimensions):                                                                         #za svaku dimenziju
            r1 = random.random()                                                                                        #dva random broja
            r2 = random.random()
            # Calculating speeds
            vel_cognitive = cp * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = cg * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social                                    #racunanje brzine po formuli

    # Izracunavanje novog polozaja cestice
    def update_position(self):
        for i in range(0, self.num_dimensions):                                                                         #za svaku dimenziju
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]                                                #sabiraju se trenutni polozaj i nova brzina, dobija se novi polozaj

            # adjust maximum position if necessary
            #   if self.position_i[i]>bounds[i][1]:
            #      self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            #   if self.position_i[i] < bounds[i][0]:
            #       self.position_i[i]=bounds[i][0]
#[]

#KLASA PSO
class PSO():
    def __init__(self, costFunc, num_dimensions, options):

        fitness_best_g = -1                                                                                             #najbolja prilagodjenost u populaciji
        pos_best_g = []                                                                                                 #najbolja pozicija u populaciji

        maxiter = options.niter
        num_particles = options.npart
        population = []
        if ((~np.isnan(options.initpopulation)).all()):
            b = np.shape(options.initpopulation)
            if (np.size(b) == 1):
                pno = b[0]
                pdim = 1
            if (pno != options.npart) or (pdim != options.nvar):
                raise Error("The format of initial population is inconsistent with desired population")
            population = options.initpopulation;
        else:
            for i in range(0, num_particles):
                x0 = (np.random.rand(num_dimensions, 1) - 0.5) * 2 * options.initspan + options.initoffset
                population.append(Particle(x0, num_dimensions, options))

        #################################
        ### The main loop ###############
        #################################
        # Begin optimization loop
        i = 0
        while i < maxiter:                                                                                              #kriterijum zaustavljanje je broj iteracija
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):                                                                            #za svaku cesticu se izracunava prilagodjenost i vrsi update lokalnih vrednosti
                population[j].evaluate(costFunc)

                # Calculating new globally best values (globally)                                                       #vrsi se i update globalnih vrednosti
                if population[j].fitness_i < fitness_best_g or fitness_best_g == -1:
                    pos_best_g = list(population[j].position_i)
                    fitness_best_g = float(population[j].fitness_i)

            #Izracunavanje nove brzine i pozicije cestice
            for j in range(0, num_particles):
                population[j].update_velocity(pos_best_g, maxiter, i, options)
                population[j].update_position()
            i += 1

        # print final results
        print('Optimal point:')
        print(pos_best_g)
        print('Optimal value:')
        print(fitness_best_g)
#[]



#GRADIJENTNI ALGORITMI

#VANILA SD
def steepest_descent(gradf, x0, gamma, epsilon, N):
    x = np.array(x0).reshape(len(x0), 1)
    for k in range(N):
        g = gradf(x)
        x = x - gamma*g
        if np.linalg.norm(g) < epsilon:
            break
    return x

#VANILA SD_V
def steepest_descent_v(gradf, x0, gamma, epsilon, N):
    x = [np.array(x0).reshape(len(x0), 1)]
    for k in range(N):
        g = gradf(x[-1])
        x.append(x[-1] - gamma*g)
        if np.linalg.norm(g) < epsilon:
            break
    return x
#[]

#SD SA MOMENTOM
def steepest_descent_with_momentum_v(gradf, x0, gamma, epsilon, omega, N):
    x = [np.array(x0).reshape(len(x0), 1)]
    v = np.zeros(shape=x[-1].shape)
    for k in range(N):
        g = gradf(x[-1])
        v = omega*v + gamma*g
        x.append(x[-1] - v)
        # U ovom algoritmu smislenije je proveravati duzinu
        # koraka (skoka) `v`, umesto samog gradijenta `g`.
        if np.linalg.norm(g) < epsilon:
            break
    return x
#[]

#SD NESTEROVA
def nesterov_gradient_descent_v(gradf, x0, gamma, epsilon, omega, N):
    x = [np.array(x0).reshape(len(x0), 1)]
    v = np.zeros(shape=x[-1].shape)
    for k in range(N):
        xpre = x[-1] - omega*v   # x_k prim
        g = gradf(xpre)
        v = omega*v + gamma*g
        x.append(x[-1] - v)
        if np.linalg.norm(g) < epsilon:
            break
    return x
#[]

#ADAGRAD
def adagrad_v(gradf, x0, gamma, epsilon1, epsilon, N):
    x = [np.array(x0).reshape(len(x0), 1)]
    v = np.zeros(shape=x[-1].shape)
    G = [np.zeros(shape=x[-1].shape)]
    for k in range(N):
        g = np.asarray(gradf(x[-1]))
        G.append(G[-1] + np.multiply(g, g))
        v = gamma * np.ones(shape=G[-1].shape)/np.sqrt(G[-1] + epsilon1) * g
        x.append(x[-1] - v)
        if np.linalg.norm(g) < epsilon:
            break
    return x, G
#[]

#RMSProp
def rmsprop_v(gradf, x0, gamma, omega, epsilon1, epsilon, N):
    x = [np.array(x0).reshape(len(x0), 1)]
    v = np.zeros(shape=x[-1].shape)
    G = [np.zeros(shape=x[-1].shape)]
    for k in range(N):
        g = np.asarray(gradf(x[-1]))
        G.append(omega*G[-1] + (1-omega)*np.multiply(g, g))
        v = gamma * np.ones(shape=g.shape)/np.sqrt(G[-1] + epsilon1) * g
        x.append(x[-1] - v)
        if np.linalg.norm(g) < epsilon:
            break
    return x, G
#[]

#ADAM
def adam_v(gradf, x0, gamma, omega1, omega2, epsilon1, epsilon, N):
    x = [np.array(x0).reshape(len(x0), 1)]
    v = [np.ones(shape=x[-1].shape)]
    m = [np.ones(shape=x[-1].shape)]
    for k in range(N):
        g = np.asarray(gradf(x[-1]))
        m.append(omega1*m[-1] + (1-omega1)*g)
        v.append(omega2*v[-1] + (1-omega2)*np.multiply(g, g))
        hat_v = np.abs(v[-1]/(1-omega2)) # abs je neophodan zbog numerickih problema kada v padne blizu nule!
        hat_m = m[-1]/(1-omega1)
        x.append(x[-1] - gamma * np.ones(shape=g.shape)/np.sqrt(hat_v + epsilon1) * hat_m)
        # print(gamma * np.ones(shape=g.shape)/np.sqrt(hat_v + epsilon1) * hat_m)
        # print(x[-1])
        if np.linalg.norm(g) < epsilon:
            break
    return x, v, m
#[]