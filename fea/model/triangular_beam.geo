//+
Point(1) = {0, 0, 0, 1};
//+
Point(2) = {2, 0, 0, 1};
//+
Point(3) = {2, 2, 0, 1};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {1, 3};
//+
Curve Loop(1) = {3, -2, -1};
//+
Plane Surface(1) = {1};
//+
Physical Curve("load", 4) = {1};
//+
Physical Curve("fixed", 5) = {2};
//+
Physical Curve("zero", 6) = {3};
//+
Physical Surface(7) = {1};
//+
Field[1] = Max;
//+
Delete Field [1];
//+
Field[1] = Cylinder;
