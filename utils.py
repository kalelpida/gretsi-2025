import numpy as np

def construit(liste):
    # construction d'une matrice d'adjacence bloc diagonale
    # matrice d'adjacence associée à un graphe composé de clusters connexes.
    dim=sum(liste)
    mat=np.zeros((dim, dim))
    s=liste[0]
    for elt in liste[1:]:
        mat[s:s+elt, s:s+elt] = np.ones((elt, elt))
        s+=elt
    return mat

def labelblocs(liste):
    #création de la liste des labels à partir d'une liste indiquant la taille des blocs de la matrice d'adjacence
    return np.concatenate([[i]*x for i, x in enumerate(liste)]).astype(int)

def zlabel(labels):
    #construction d'une matrice indicatrice de cluster selon le label de chaque nœud
    return np.array([(labels==elt)*1 for elt in set(labels)])

def zlabel2(labels):
    #construction d'une matrice d'adjacence partitionnée. Les nœuds appartenant à une même classe (même label) sont tous connectés.
    z=zlabel(labels)
    return np.dot(z.T, z)

def groupeSol(sol):
    # sol : liste des indices des classes pour chaque nœud
    # **renvoie une liste des nœud appartenant à une même classe 
    return [np.where(sol==elt)[0] for elt in np.unique(sol)]


def pmats(N, p, q):
    # crée des matrices de probabilités de présence d'arête entre deux deux.
    # N: nombre de nœuds
    # p: probabilités de liaison de chaque nœud, en supposant qu'il est théoriquement relié à l'autre
    # q: probabilités de liaison de chaque nœud, en supposant qu'il n'est théoriquement pas relié à l'autre
    # renvoie les matrices de probabilité de présence d'un lien entre deux nœuds. La première suppose qu'ils ont théoriquement reliés, la seconde non.
    if isinstance(p, float):
        p=np.full(N, p)
    if isinstance(q, float):
        q=np.full(N, q)
    #assert p.shape==q.shape
    return (p+p[:,np.newaxis])*0.5, (q+q[:,np.newaxis])*0.5

def pmat_labels(labels, pmat1, pmat0):
    #Calcul des paramètres de la loi de Bernoulli de chaque arête de la matrice d'adjacence.
    # labels : indices des classes associées à chaque nœud
    # pmat1: matrice des probabilités de présence d'une arête entre deux nœuds supposés liés
    # pmat0: matrice des probabilités de présence d'une arête entre deux nœuds supposés non liés
    # ** calcule une matrice de probabilité de présence d'un lien entre deux nœud.
    v = zlabel2(labels)
    return pmat1*v+pmat0*(1-v)

def brmat(paramat, graine=None):
    #"bruite" une matrice d'adjacence 
    # paramat : paramètres des lois de Bernoulli associées à la présence d'une arête entre chaque paire de nœuds.
    # ** renvoie une matrice d'adjacence symétrique simulée
    n, m = paramat.shape
    assert n==m
    rng = np.random.default_rng(graine)
    matb=np.empty((n,n))
    params = paramat[np.triu_indices(n)]
    gens = rng.binomial(1, params, size=round(0.5*n*(n+1)))
    matb[np.triu_indices(n)] = gens
    matb[np.tril_indices(n, k=-1)] = np.transpose(matb)[np.tril_indices(n, k=-1)]
    return matb



def modularite(labels, mat, *args):
    #implémentation de la modularité de Newman-Girvan
    # labels: indices des classes associées à chaque nœud
    # mat: matrice d'adjacence observée
    # args:  non utilisés
    # ** calcul de la modularité
    ntot=len(labels)
    ens=set(labels)
    z=np.empty((len(ens),ntot))
    for i, elt in enumerate(ens):
        z[i] = (labels==elt)*1
    calcs=np.dot(z, np.dot(mat, z.T))
    total=np.sum(calcs)#nombre total d'arêtes
    eii = np.diag(calcs)/total
    ai = np.sum(calcs, axis=1)/total
    return np.sum(eii-ai**2)

def vraiss(labels, mat, paramat_lies, paramat_nlies):
    #Calcule la log-vraisemblance de $labels étant donnée une observation $mat et des probabilités paramat.
    # labels: indices des classes associées à chaque nœud
    # mat : matrice d'adjacence observée
    # paramat_lies: matrice des probabilités de présence d'une arête entre deux nœuds supposés liés
    # paramat_nlies: matrice des probabilités de présence d'une arête entre deux nœuds supposés non liés
    # ** renvoie : valeur de log-vraisemblance
    return vraissz(np.array([(labels==elt)*1 for elt in set(labels)]), mat, paramat_lies, paramat_nlies)

def vraissz(z, mat, paramat_lies, paramat_nlies):
    # idem ci-dessus
    # z: matrice indicatrice d'appartenance d'un nœud à un cluster z[k,n]=1 si nœud n appartient à cluster k
    # ** renvoie : valeur de log-vraisemblance
    clusmat=np.dot(z.T, z)
    aclusmat=1-clusmat
    amat=1-mat
    return np.sum(np.log(paramat_lies)*mat*clusmat+np.log(paramat_nlies)*mat*aclusmat+np.log(1-paramat_lies)*amat*clusmat+np.log(1-paramat_nlies)*amat*aclusmat)


def gibbsSampler(params, objfonc, nit=100):
    # implémentation d'un gibbsSampler
    # params :
    #   - labels : indices du cluster de chaque nœud
    #   - matrice d'adjacence observée
    #   - autres paramètres tels que les 
    # objfonc : une fonction objectif à maximiser, i.e. une log-vraisemblance
    # ** renvoie : meilleure liste de labels trouvée maximisant objfonc, la meilleure valeur trouvée, la valeur initiale

    rng=np.random.default_rng(1234)
    #if 'vrais' in objfonc.__name__:
    #    #Ne semble pas plus rapide en pratique
    #    return gibbsSamplerVraiss(*params, nit=nit)
    
    labels=params[0]
    labels-=min(labels)
    K=len(set(labels))
    N=len(labels)
    best_labels = labels.copy()
    flabels = objfonc(*params)
    best_objective = flabels

    # Gibbs sampling
    poids_jposs=np.empty(K)
    for iteration in range(nit):
        for j in range(N):
            for k in range(K):
                labels[j]=k
                poids_jposs[k] = objfonc(labels, *params[1:])
            exppoids_jposs_norm = np.exp(poids_jposs- max(poids_jposs))#normalisation par max
            ps = exppoids_jposs_norm/sum(exppoids_jposs_norm)
            #choix k
            kchoisi = rng.choice(K, p=ps)
            labels[j] = kchoisi
            objective = poids_jposs[kchoisi]

            if np.any(poids_jposs > best_objective):
                best_objective = np.max(poids_jposs)
                best_labels= labels.copy()
                best_labels[j] = np.argmax(poids_jposs)
            
    return best_labels , best_objective, flabels 

def gibbsSamplerVraiss(labels, A, pmat1, pmat0, nit=100):
    #gibbs sampler, moins de calculs en théorie.
    # labels : solution initiale, indices du cluster d'appartenance de chaque nœud
    # A : matrice d'adjacence observée
    # pmat1: matrice des probabilités de présence d'une arête entre deux nœuds supposés liés
    # pmat0: matrice des probabilités de présence d'une arête entre deux nœuds supposés non liés
    rng=np.random.default_rng(1234)

    Z=np.array([labels==elt for elt in set(labels)])
    K, N = Z.shape  # Size of matrix B
    pm1log= np.log(pmat1)
    cpm1log=np.log(1-pmat1)
    pm0log= np.log(pmat0)
    cpm0log=np.log(1-pmat0)
    # Variables to track the best configuration
    best_Z = Z.copy()
    flabels = vraissz(Z, A, pmat1, pmat0)
    best_objective = flabels

    # Gibbs sampling
    sansj = np.eye(N)==0
    zj_poss = np.eye(K)
    poids_jposs=np.empty(K)
    for iteration in range(nit):
        for j in range(N):
            for k in range(K):
                poids_jposs[k] = np.dot(pm1log[j,sansj[j]],Z[k, sansj[j]]*A[k, sansj[j]]) + np.dot(cpm1log[j,sansj[j]],(1-Z[k, sansj[j]])*A[k, sansj[j]]) + np.dot(pm0log[j,sansj[j]],Z[k, sansj[j]]*(1-A[k, sansj[j]])) + np.dot(cpm0log[j,sansj[j]],(1-Z[k, sansj[j]])*(1-A[k, sansj[j]]))
            exppoids_jposs_norm = np.exp(poids_jposs- max(poids_jposs))#normalisation par max
            ps = exppoids_jposs_norm/sum(exppoids_jposs_norm)
            Z[:,j] = zj_poss[rng.choice(K, p=ps)]
            objective = vraissz(Z, A, pmat1, pmat0)

            if objective > best_objective:
                best_objective = objective
                best_Z = Z.copy()

    best_labels = np.argmax(best_Z, axis=0)
    return best_labels , best_objective, flabels 
