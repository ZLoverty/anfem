SetFactory("OpenCASCADE");


// Parameters
W_TOTAL = 200;
H_TOTAL = 80;
h = 6;
m = 3;
l = 2 * h;
N = 10;
w = 2 * h;
cl__1 = 6; // outer boundary characteristic length
cl__2 = 2; // channel characteristic length

// Draw the pool4
Point(1) = {0, 0, 0, cl__1} ;
Point(2) = {W_TOTAL, 0, 0, cl__1} ;
Point(3) = {W_TOTAL, H_TOTAL, 0, cl__1} ;
Point(4) = {0, H_TOTAL, 0, cl__1} ;

Line(1) = {1, 2};  // bottom
Line(2) = {2, 3};  // right
Line(3) = {3, 4};  // top
Line(4) = {4, 1};  // left

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Periodic boundary conditions
// Map: right -> left
Periodic Curve {2} = {4} Translate {W_TOTAL, 0, 0};
// Map: top -> bottom
Periodic Curve {3} = {1} Translate {0, H_TOTAL, 0};

// determine x01, y01, the starting point of bottom ratchet
x01 = 0.5 * (W_TOTAL - N*l - 2*m);
y01 = 0.5 * (H_TOTAL - 4*h - w);

Point(5) = {x01, y01, 0, cl__2};
Point(6) = {x01, y01 + 2*h, 0, cl__2};

p = 7;

// Loop over teeth
For i In {0:N-1}
    // Ratchet tooth corner
    Point(p)   = {x01 + m + i*l, y01 + 2*h, 0, cl__2};
    Point(p+1) = {x01 + m + i*l, y01 + h, 0, cl__2};
    p += 2;
EndFor

Point(p)   = {x01 + m + N*l, y01 + 2*h, 0, cl__2};
Point(p+1) = {x01 + 2*m + i*l, y01 + 2*h, 0, cl__2};
Point(p+2) = {x01 + 2*m + i*l, y01, 0, cl__2};

// last point number of the first half
p_last_1 = 4 + 2*N + 5;

For i In {5: p_last_1-1}
    Line(i) = {i, i+1};
EndFor

Line(p_last_1) = {p_last_1, 5};

Line Loop(2) = {5: p_last_1};
Plane Surface(2) = {2};

// Subtract circle from rectangle
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }



// Top part of the channel

// determine x02, y02, the starting point of bottom ratchet
x02 = 0.5 * (W_TOTAL - N*l - 2*m);
y02 = H_TOTAL - 0.5 * (H_TOTAL - 4*h - w);

p_first_2 = p_last_1 + 10;
Point(p_first_2) = {x02, y02, 0, cl__2};
Point(p_first_2 + 1) = {x02, y02 - 2*h, 0, cl__2};

p = p_first_2 + 2;

// Loop over teeth
For i In {0:N-1}
    // Ratchet tooth corner
    Point(p)   = {x02 + m + i*l, y02 - 2*h, 0, cl__2};
    Point(p+1) = {x02 + m + i*l, y02 - h, 0, cl__2};
    p += 2;
EndFor

Point(p)   = {x02 + m + N*l, y02 - 2*h, 0, cl__2};
Point(p+1) = {x02 + 2*m + i*l, y02 - 2*h, 0, cl__2};
Point(p+2) = {x02 + 2*m + i*l, y02, 0, cl__2};

// last point number of the first half
p_last_2 = p_first_2 + 2*N + 5 - 1;

For i In {p_first_2: p_last_2-1}
    Line(i) = {i, i+1};
EndFor

Line(p_last_2) = {p_last_2, p_first_2};

Line Loop(3) = {p_first_2: p_last_2};
Plane Surface(3) = {3};

// Subtract circle from rectangle
BooleanDifference{ Surface{1}; Delete; }{ Surface{3}; Delete; }

// Define physical groups (optional)
Physical Surface("domain", 1) = {1};
a[] = {5: p_last_1};
b[] = {p_first_2: p_last_2};
all[] = {};
all[] += a[];
all[] += b[];
Physical Line("wall", 3) = {all[]};

/*
L = 10;          // Total length of the channel
H = 2;           // Total height
N = 3;           // Number of ratchet teeth
tooth_L = L / N; // Length of one tooth
tooth_H = 0.5;   // Tooth height

// Starting points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, H, 0, 1.0};
l_cnt = 1;
Line(l_cnt) = {1, 2}; // left wall
l_cnt += 1;

p = 3;
last_top = 2;

// Loop over teeth
For i In {1:N}
    x0 = i * tooth_L;
    // Ratchet tooth corner
    Point(p)   = {x0, H - tooth_H, 0, 1.0};
    Point(p+1) = {x0, H, 0, 1.0};

    Line(l_cnt) = {last_top, p}; // slope down
    l_cnt += 1;
    Line(l_cnt) = {p, p+1};      // vertical rise
    l_cnt += 1;

    last_top = p+1;
    p += 2;
EndFor

// Right top corner and close the loop
Point(p) = {L + 0.5, H, 0, 1.0};
Line(l_cnt) = {last_top, p};
l_cnt += 1;

Point(p+1) = {L + 0.5, 0, 0, 1.0};
Line(l_cnt) = {p, p+1}; // right wall
l_cnt += 1;

Line(l_cnt) = {p+1, 1}; // bottom wall
l_cnt += 1;

// Create curve loop
cl[] = {};
For i In {1:(l_cnt - 1)}
    cl[i - 1] = i;
EndFor

Line Loop(1) = {cl[]};
Plane Surface(1) = {1};

// Optional physical surface
Physical Surface("RatchetChannel") = {1};
*/