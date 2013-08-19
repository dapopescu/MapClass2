#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <list>
#include <unordered_map>
#include <time.h>
#include "MapBeamLine.h"

using namespace std;
using namespace boost;

static string drift = "DRIFT";
static string quadrupole = "QUADRUPOLE";
static string sbend = "SBEND";
static string multipole = "MULTIPOLE";
static string decapole = "DECAPOLE";
static string sextupole = "SEXTUPOLE";
static string octupole = "OCTUPOLE";
static string DX = "DX";
static string DY = "DY";


MapBeamLine::MapBeamLine(Twiss t, int order, int nbthreads, int fmultipole, bool strpl) {
	omp_set_num_threads(nbthreads); 
	if (strpl)
		t.stripLine();
	vector<vector<Polynom<double>>> v = separateComplexList(EQ(4, order));
	Polynom<double> x = X<double>(order);
	Polynom<double> px = PX<double>(order);
	Polynom<double> y = Y<double>(order);
	Polynom<double> py = PY<double>(order);
	Polynom<double> d = D<double>(order);
	Polynom<double> s = S<double>(order);
	Polmap<double> R = generateDefaultMap(x, px, y, py, d, s);
	
	Polmap<double>* Res = new Polmap<double>[nbthreads];
	for (int i = 0; i < nbthreads; i ++)
		Res[i] = R; 
	int size = t.elems.size();	
	Polmap<double>* mp = new Polmap<double>[size];
	#pragma omp parallel for shared(Res) schedule(dynamic, 10)
        for (int i = 0; i < size; i ++) 
        	mp[i] = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole); 
                         
               

	if (strpl) {
		double start = omp_get_wtime();
		#pragma omp parallel for shared(Res) schedule(static)
		
		for (int i = 0; i < size; i ++) {
			int index = omp_get_thread_num();
			Res[index] =  mp[i] * Res[index];		
		}
		double end = omp_get_wtime();
		cout << 1000 * (end - start) << endl; 
	}
	else {
		#pragma omp parallel for shared(Res) schedule(static)
                for (int i = 0; i < size; i ++) {
                        int index = omp_get_thread_num();
                        Polmap<double> mp = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole);
                        if (mp.pols.size() != 0)
				Res[index] = mp * Res[index];
                }
	}
	double start = omp_get_wtime(); 
	
	if (nbthreads >= 32) {
		Polmap<double>* Res0 = new Polmap<double>[16];
		for (int i = 0; i < 16; i ++)
                        Res0[i] = R;
                #pragma omp parallel for shared(Res) schedule(static) num_threads(16)
                for (int i = 0; i < nbthreads; i ++) {
                        int index =  omp_get_thread_num();
                        Res0[index] = Res[i] * Res0[index];

                }


		Polmap<double>* Res1 = new Polmap<double>[8];
		for (int i = 0; i < 8; i ++)
			Res1[i] = R;
		 #pragma omp parallel for shared(Res) schedule(static) num_threads(8)
                for (int i = 0; i < 8; i ++) {
                        int index =  omp_get_thread_num();
                        Res1[index] = Res0[i] * Res1[index];

                }

		Polmap<double>* Res2 = new Polmap<double>[4];
		for (int i = 0; i < 4; i ++) 
			Res2[i] = R;
		//double start2 = omp_get_wtime();

		#pragma omp parallel for shared(Res) schedule(static) num_threads(4)
		for (int i = 0; i < 4; i ++) {
			int index =  omp_get_thread_num();
			Res2[index] = Res1[i] * Res2[index];
	
		}
		Polmap<double>* Res3 = new Polmap<double>[2];
		Res3[0] = R;
		Res3[1] = R;
		#pragma omp parallel for shared(Res) schedule(static) num_threads(2)
                for (int i = 0; i < 4; i ++) {
                        int index =  omp_get_thread_num();
                        Res3[index] = Res2[i] * Res3[index];

                }

		//double end2 = omp_get_wtime();
		//cout << 1000 * (end2 - start2) << endl;
		//double start3 = omp_get_wtime();

		R = Res3[1] * Res3[0];
		//double end3 = omp_get_wtime();
		//cout << 1000 * (end3 - start3) << endl;
	}
	else {
		R = Res[0];
		for (int i = 1; i < nbthreads; i ++) {
			  R = Res[i] * R;
		}
	}
	double end = omp_get_wtime();
	cout << 1000 *(end - start) << endl;
	polmap = R.getMap();
	for (unordered_map<string, Polynom<double>>:: iterator it = R.pols.begin(); it != R.pols.end(); it ++) 
		pols[it->first] = it->second; 
	delete [] Res;
	delete [] mp;
}

MapBeamLine::MapBeamLine(Twiss t, Twiss terr, int order, int nbthreads, int fmultipole, bool strpl) {
	omp_set_num_threads(nbthreads);
	if (strpl) {
		t.stripLine();
		terr.stripLine();
	}
	vector<vector<Polynom<double>>> v = separateComplexList(EQ(4, order));
	Polynom<double> x = X<double>(order);
	Polynom<double> px = PX<double>(order);
	Polynom<double> y = Y<double>(order);
	Polynom<double> py = PY<double>(order);
	Polynom<double> d = D<double>(order);
	Polynom<double> s = S<double>(order);
	Polmap<double> R = generateDefaultMap(x, px, y, py, d, s);
	
	Polmap<double>* Res = new Polmap<double>[nbthreads];
	for (int i = 0; i < nbthreads; i ++)
		Res[i] = R; 
	
	int size = t.elems.size();
	if (strpl) {	
		#pragma omp parallel for shared(Res) schedule(static)
		for (int i = 0; i < size; i ++) {
			int index = omp_get_thread_num();
			Polmap<double> mp = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole);	
			double dx = atof(terr.elems[i][DX].c_str());
          		double dy = atof(terr.elems[i][DY].c_str());
			mp = mp.eval("x", Polynom<double>(order, 1E-18, "x", 1) + dx); 
			mp = mp.eval("y", Polynom<double>(order, 1E-18, "y", 1) + dy);
			Res[index] = mp * Res[index];		
		}
	} 
	else {
		#pragma omp parallel for shared(Res) schedule(static)
                for (int i = 0; i < size; i ++) {
                        int index = omp_get_thread_num();
                        Polmap<double> mp = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole);
                        double dx = atof(terr.elems[i][DX].c_str());
                        double dy = atof(terr.elems[i][DY].c_str());
                        mp = mp.eval("x", Polynom<double>(order, 1E-18, "x", 1) + dx);
                        mp = mp.eval("y", Polynom<double>(order, 1E-18, "y", 1) + dy);
                        if (mp.pols.size() != 0)
				Res[index] = mp * Res[index];
                }

	}
	R = Res[0];
	for (int i = 1; i < nbthreads; i ++)
		R = Res[i] * R;
	polmap = R.getMap();
	for (unordered_map<string, Polynom<double>>:: iterator it = R.pols.begin(); it != R.pols.end(); it ++) 
		pols[it->first] = it->second; 
	delete [] Res;
}

MapBeamLine::MapBeamLine(string filename, int order, int nbthreads, int fmultipole, bool strpl) {
	Twiss t = Twiss(filename);
	if (strpl)
		t.stripLine();
	omp_set_num_threads(nbthreads);
	vector<vector<Polynom<double>>> v = separateComplexList(EQ(4, order));
	Polynom<double> x = X<double>(order);
	Polynom<double> px = PX<double>(order);
	Polynom<double> y = Y<double>(order);
	Polynom<double> py = PY<double>(order);
	Polynom<double> d = D<double>(order);
	Polynom<double> s = S<double>(order);
	Polmap<double> R = generateDefaultMap( x, px, y, py, d, s);
	
	Polmap<double>* Res = new Polmap<double>[nbthreads];
	for (int i = 0; i < nbthreads; i ++)
		Res[i] = R; 
	
	int size = t.elems.size();
	if (strpl) {	
		#pragma omp parallel for shared(Res) schedule(static)
		for (int i = 0; i < size; i ++) {
			int index = omp_get_thread_num();
			Polmap<double> mp = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole);	
			Res[index] = mp * Res[index];		
		}
	}
	else {
		#pragma omp parallel for shared(Res) schedule(static)
                for (int i = 0; i < size; i ++) {
                        int index = omp_get_thread_num();
                        Polmap<double> mp = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole);
                        if (mp.pols.size() != 0)
				Res[index] = mp * Res[index];
                }

	}
	R = Res[0];
	for (int i = 1; i < nbthreads; i ++)
		R = Res[i] * R;
	polmap = R.getMap();
	for (unordered_map<string, Polynom<double>>:: iterator it = R.pols.begin(); it != R.pols.end(); it ++)
		pols[it->first] = it->second; 
	delete [] Res; 
}

MapBeamLine::MapBeamLine(string filename, string filenameerr, int order, int nbthreads, int fmultipole, bool strpl) {
	Twiss t = Twiss(filename);
	Twiss terr = Twiss(filenameerr);
	if (strpl) {
		t.stripLine();
		terr.stripLine();
	}
	omp_set_num_threads(nbthreads);
	vector<vector<Polynom<double>>> v = separateComplexList(EQ(4, order));
	Polynom<double> x = X<double>(order);
	Polynom<double> px = PX<double>(order);
	Polynom<double> y = Y<double>(order);
	Polynom<double> py = PY<double>(order);
	Polynom<double> d = D<double>(order);
	Polynom<double> s = S<double>(order);
	Polmap<double> R = generateDefaultMap(x, px, y, py, d, s);
	
	Polmap<double>* Res = new Polmap<double>[nbthreads];
	for (int i = 0; i < nbthreads; i ++)
		Res[i] = R; 
	
	int size = t.elems.size();
	if (strpl) {	
		#pragma omp parallel for shared(Res) schedule(static)
		for (int i = 0; i < size; i ++) {
			int index = omp_get_thread_num();
			Polmap<double> mp = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole);	
			double dx = atof(terr.elems[i][DX].c_str());
          		double dy = atof(terr.elems[i][DY].c_str());
			mp = mp.eval("x", Polynom<double>(order, 1E-18, "x", 1) + dx); 
			mp = mp.eval("y", Polynom<double>(order, 1E-18, "y", 1) + dy);
			Res[index] = mp * Res[index];		
		}
	}
	else {
		#pragma omp parallel for shared(Res) schedule(static)
                for (int i = 0; i < size; i ++) {
                        int index = omp_get_thread_num();
                        Polmap<double> mp = mapForElement(t.elems[i], v, x, px, y, py, d, s, fmultipole);
                        double dx = atof(terr.elems[i][DX].c_str());
                        double dy = atof(terr.elems[i][DY].c_str());
                        mp = mp.eval("x", Polynom<double>(order, 1E-18, "x", 1) + dx);
                        mp = mp.eval("y", Polynom<double>(order, 1E-18, "y", 1) + dy);
                        if (mp.pols.size() != 0)
				Res[index] = mp * Res[index];
                }

	}
	R = Res[0];
	for (int i = 1; i < nbthreads; i ++)
		R = Res[i] * R;
	polmap = R.getMap();
	for (unordered_map<string, Polynom<double>>:: iterator it = R.pols.begin(); it != R.pols.end(); it ++) 
		pols[it->first] = it->second; 
	delete [] Res; 
}


int main(int argc,char *argv[]) {


        int order = 3;
        int nbthreads = 1;
        string filename = "";
        for (int i = 1; i < argc; i += 2) {
                if (strcmp(argv[i], "-o") == 0)
                        order = atoi(argv[i + 1]);
                else if (strcmp(argv[i], "-n") == 0)
                                nbthreads = atoi(argv[i + 1]);
                else if (strcmp(argv[i], "-f") == 0)
                                filename = argv[i + 1];
        }
        Twiss t = Twiss(filename);

        //double start = omp_get_wtime();
        MapBeamLine mbl = MapBeamLine(t, order, nbthreads, 0, true);


        //double end = omp_get_wtime();
        //cout << "running time = " <<  end - start << endl;
        //mbl.printpolmap();
        return 0;
}


