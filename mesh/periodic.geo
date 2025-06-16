// Mesh size
cl = 0.5;

// Define 4 corner points
Point(1) = {0, 0, 0, cl};
Point(2) = {10, 0, 0, cl};
Point(3) = {10, 10, 0, cl};
Point(4) = {0, 10, 0, cl};

// Define edges (curves)
Line(1) = {1, 2};  // bottom
Line(2) = {2, 3};  // right
Line(3) = {3, 4};  // top
Line(4) = {4, 1};  // left

// Line loop and surface
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// Define physical groups (optional)
Physical Surface("domain", 1) = {6};
Physical Curve("left", 2) = {4};
Physical Curve("right", 2) = {2};
Physical Curve("bottom", 4) = {1}; 
Physical Curve("top", 5) = {3};

// Periodic boundary conditions
// Map: right -> left
Periodic Curve {2} = {4} Translate {1, 0, 0};
// Map: top -> bottom
Periodic Curve {3} = {1} Translate {0, 1, 0};
