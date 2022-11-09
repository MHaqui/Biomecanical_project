% Calcul de commandabilité et d'observabilité
% Mettre à jour les valeurs des paramètres
M1 = 0.5 % kg
M2 = 1 %kg
L1 = 0.02 %m
L2 = 0.03 %m
g = 9.8 %m.s^-2

A = [
    0 0 1 0;
    0 0 0 1;
    ((M2+M1)*g)/(L1*M1) (M2*g)/(L1*M2) 0 0;
    ((M2+M1)*g)/(L2*M1) ((M2-M1)*g)/(L2*M1) 0 0;
    ]
B = [
    0;
    0;
    0;
    0;
    ]

C = [0 0 0 1]

Co = [C; C*A; C*A^2;C*A^3]
eig(Co)