#include <stdio.h>
#include <math.h>
//#include "conio.h"
#include <string.h>
#include <malloc.h>

#define PI (3.1415926535897932384626433832795)
#define M_2PI (2.0*PI)
#define rad (PI/180.0)
#define VC  (299792458.0) //Скорость света в м/с
#define mVC (0.149896229) // [км/мкс]

int M0U=3;//Стпень полинома вписываемого в зависимость максимума ДН от частоты
double KAmp0[4]={0.0};//коэффициенты полинома для волны Н0
double KAmpU[4]={0.0};//коэффициенты полинома для суммы волн
double alfa=0.0;
int Mstr=4;
double Kstr[5]={0.0};
double DEP=0.281600;

double JD(int Y,int M,int DT){
	int a,b=0,m,y;
	long c;

	y=Y;
	m=M;
	if(M<3){
		y--;
		m+=12;
	}
	a=y/100;
	if(Y+M/100+DT/10000>1582.1015)	b+=2-a+(int)(a/4);
	c=365.25*y;
	if(y<0)	c=365.25*y-0.75;
	return c+(long)(30.6001*(m+1))+DT+1720994.5+b;
}

void S0TIM(double MJDD, double *STG0, double *DTETA, double *RKPRIM){
	double TU;
	TU=(MJDD-51544.5)/36525.0;
	*STG0=8640184.0*TU/86400.0;
	*STG0=*STG0-(int)(*STG0);
	*STG0=*STG0+24110.54841/86400.0+0.812866*TU/86400.0+(0.093104*TU*TU-(6.2E-6)*TU*TU*TU)/86400.0;
	*STG0=*STG0-(int)(*STG0);
	if(*STG0<0) *STG0=1.0+*STG0;
	if(*STG0>=0){
		*DTETA=1.002737909350795+(5.9006E-11)*TU-(5.9E-15)*TU*TU;
		*RKPRIM=0.997269566329084-(5.8684E-11)*TU+(5.9E-15)*TU*TU;
	}
}

void GetCurentST(int Year,int Month,int Day,double *ST,double *DTETA){
	int Ijd;
	double jd,mjd,dteta,DKSI,DEPS,EPS,S0;
	double RKPRIM;

	jd=JD(Year,Month,Day);//Юлианская Дата начала расчета
	mjd=jd - 2400000.5;//Модифицированная Юлианская Дата начала расчета
	Ijd=(int)(mjd);//Целая часть ЮД
	//-----------------------------------------------//
	S0TIM((double)(Ijd),&S0,&dteta,&RKPRIM);//Cреднее звездное время
	//-----------------------------------------------//
	*ST=S0;
	*DTETA=dteta;
}

void TimeToo(double Dd,int *h,int *m,double *s){
	int th,tm;
	double t,ts;
	t=Dd*86400.0;
	if(t<0.0) t+=86400.0;
	th=(int)(t/3600.0);
	tm=(int)((t-(double)(th)*3600.0)/60.0);
	ts=t-(double)(th)*3600.0-(double)(tm)*60.0;
	printf("th=%d tm=%d ts=%lf\n",th,tm,ts);
	*h=th;
	*m=tm;
	*s=ts;
}

void Ll360(double *Aa){
	double Bb;
	Bb=*Aa;
	//printf("B=%lf\n");
	if(Bb!=0.0) Bb=Bb-(double)((int)(Bb/PI/2))*PI*2.0;
	//while(B<0) B=B+PI*2;
	*Aa=Bb;
}

void Topoc_to_Ant(double el,double az,double *gam,double *ep){
	double Az,El,gam0;
	double Ua=7.0*rad;
	double Ub=10.0*rad;
	
	Az=az+Ua;
	El=el;
	*ep=asin(-cos(Az)*cos(El));
	gam0=Ub-atan2(-sin(Az)*cos(El),sin(El));
	*gam = gam0;
//	*gam=asin(sin(gam0)*cos(*ep));  # Для экспериментов по когер. эхо
}

void Ant_to_Topoc(double ep,double gam,double *el,double *az){
	double gam0;
	double Ua=7.0*rad;
	double Ub=10.0*rad;

//	gam0=asin(sin(gam)/cos(ep));  # Для экспериментов по когер. эхо
    gam0 = gam;
	*el=asin(cos(Ub-gam0)*cos(ep));
	*az=1.5*PI-Ua-atan2(sin(ep),cos(ep)*sin(Ub-gam0));
}

/* Shortcut to calculate only theta (PI/2 - elevation)
 * return 0 if gam is incorrect, otherwise return 1
 */
int Theta_from_Ant(double ep, double gam, double *theta){
    double gam0;
	double Ub = 10.0*rad;
	if ((gam < (-PI/2 + Ub) | (gam > (PI/2 + Ub)))){
	    return 0;
	}else{
        double sign = copysign(1., gam) * copysign(1., cos(ep));
        double robust_div = fmin(fabs(sin(gam) / cos(ep)), 1.);
        gam0 = asin(sign * robust_div);
        *theta = PI/2. - asin(cos(Ub - gam0) * cos(ep));
        return 1;
	}
}


void Star_Radar(int year,int mes,int day,double t,double LG,double FI,double Ra,double Dec,double *EL,double *AZ,double *EP,double *GAM){
	int hour,min;
	double sec;
	double jd,mjd,QS,S0,DTETA,WE;
	double th,el,az,gam,ep,x,y;

//	TimeToo(t/86400.0,&hour,&min,&sec);
	GetCurentST(year,mes,day,&S0,&DTETA);//S0-истинное звездное время на начало дня наблюдения
//	TimeToo(S0,&hour,&min,&sec);
	WE=2.0*PI*DTETA/86400.0;//Скорость вращения Земли в рад/сек
	S0=S0*2.0*PI;//Истинное звездное время в радианах
	QS=S0+WE*t;//Текущее истинное звездное время в радианах
	Ll360(&QS);
//	TimeToo(QS/2.0/PI,&hour,&min,&sec);
	th=QS+LG-Ra;//Часовой угол
	Ll360(&th);
//	printf("th=%lf\n",th/rad);
	el=asin(sin(Dec)*sin(FI)+cos(Dec)*cos(FI)*cos(th));
	az=atan2(-sin(th)*cos(Dec),sin(Dec)*cos(FI)-cos(Dec)*sin(FI)*cos(th));
	if(az<0.0)
		az+=2.0*PI;
	Topoc_to_Ant(el,az,&gam,&ep);
	*EL=el;
	*AZ=az;
	*EP =ep;
	*GAM=gam;
}

long double Power(long double v,int p){
	if(p==0) return 1;
	if(p>1) v*=Power(v,p-1);
	return v;
}

void Calculate(int K,int pts,double x[],double y[],double a[])
{
int i,j,k;
long double s,t,M;
double b[10],sums[10][10];

	for(i=0; i<K+1; i++){
		a[i]=0.0;
		for(j=0; j<K+1; j++){
			sums[i][j] = 0.0;
	   		for(k=1; k<=pts; k++)
	    		sums[i][j]+=Power(x[k],i+j);
		}
	}

	for(i=0; i<K+1; i++){
		b[i]=0;
		for(k=1; k<=pts; k++)
			b[i] +=Power(x[k],i)*y[k];
	 }

	for(k=0; k<K+1; k++){
		for(i=k+1; i<K+1; i++){
			M=sums[i][k]/sums[k][k];
			for(j=k; j<K+1; j++)
				sums[i][j] -= M * sums[k][j];
			b[i] -= M*b[k];
		}
	}

	for(i=K;i>=0;i--){
		s=0;
		for(j = i; j<K+1; j++)
			s+=sums[i][j]*a[K-j];
		a[K-i] = (b[i]-s)/sums[i][i];
	}
}


/*************************************************************************
Интегралы Френеля.

Входные параметры:
    X   -   аргумент функции
    
Выходные параметры:
    C,S -   интегралы Френеля C(X) и S(X)

Допустимые значения:
    X - любое вещественное число

Относительная погрешность:

ФУНКЦИЯ    ОБЛАСТЬ     #ТЕСТОВ         МАКС.      СРЕДН.
S(x)         0, 10       10000       2.0e-15     3.2e-16
C(x)         0, 10       10000       1.8e-15     3.3e-16

Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
*************************************************************************/
void fresnelintegral(double x, double *c, double *s)
{
    double xxa;
    double f;
    double g;
    double cc;
    double ss;
    double t;
    double u;
    double x2;
    double sn;
    double sd;
    double cn;
    double cd;
    double fn;
    double fd;
    double gn;
    double gd;
    double mpi;
    double mpio2;

    mpi = 3.14159265358979323846;
    mpio2 = 1.57079632679489661923;
    xxa = x;
    x = fabs(xxa);
    x2 = x*x;
    if( x2<2.5625 )
    {
        t = x2*x2;
        sn = -2.99181919401019853726E3;
        sn = sn*t+7.08840045257738576863E5;
        sn = sn*t-6.29741486205862506537E7;
        sn = sn*t+2.54890880573376359104E9;
        sn = sn*t-4.42979518059697779103E10;
        sn = sn*t+3.18016297876567817986E11;
        sd = 1.00000000000000000000E0;
        sd = sd*t+2.81376268889994315696E2;
        sd = sd*t+4.55847810806532581675E4;
        sd = sd*t+5.17343888770096400730E6;
        sd = sd*t+4.19320245898111231129E8;
        sd = sd*t+2.24411795645340920940E10;
        sd = sd*t+6.07366389490084639049E11;
        cn = -4.98843114573573548651E-8;
        cn = cn*t+9.50428062829859605134E-6;
        cn = cn*t-6.45191435683965050962E-4;
        cn = cn*t+1.88843319396703850064E-2;
        cn = cn*t-2.05525900955013891793E-1;
        cn = cn*t+9.99999999999999998822E-1;
        cd = 3.99982968972495980367E-12;
        cd = cd*t+9.15439215774657478799E-10;
        cd = cd*t+1.25001862479598821474E-7;
        cd = cd*t+1.22262789024179030997E-5;
        cd = cd*t+8.68029542941784300606E-4;
        cd = cd*t+4.12142090722199792936E-2;
        cd = cd*t+1.00000000000000000118E0;
//        s = ap::sign(xxa)*x*x2*sn/sd;
//        c = ap::sign(xxa)*x*cn/cd;
        *s = x*x2*sn/sd;
        *c = x*cn/cd;

        *s = xxa*x*x2*sn/sd/fabs(xxa);
        *c = xxa*x*cn/cd/fabs(xxa);
//        *s = x*x2*sn/sd;
        //*c = x*cn/cd;

        return;
    }
    if( x>36974.0 )
    {
//        c = ap::sign(xxa)*0.5;
//        s = ap::sign(xxa)*0.5;
        *c = xxa*0.5/fabs(xxa);
        *s = xxa*0.5/fabs(xxa);
        return;
    }
    x2 = x*x;
    t = mpi*x2;
    u = 1/(t*t);
    t = 1/t;
    fn = 4.21543555043677546506E-1;
    fn = fn*u+1.43407919780758885261E-1;
    fn = fn*u+1.15220955073585758835E-2;
    fn = fn*u+3.45017939782574027900E-4;
    fn = fn*u+4.63613749287867322088E-6;
    fn = fn*u+3.05568983790257605827E-8;
    fn = fn*u+1.02304514164907233465E-10;
    fn = fn*u+1.72010743268161828879E-13;
    fn = fn*u+1.34283276233062758925E-16;
    fn = fn*u+3.76329711269987889006E-20;
    fd = 1.00000000000000000000E0;
    fd = fd*u+7.51586398353378947175E-1;
    fd = fd*u+1.16888925859191382142E-1;
    fd = fd*u+6.44051526508858611005E-3;
    fd = fd*u+1.55934409164153020873E-4;
    fd = fd*u+1.84627567348930545870E-6;
    fd = fd*u+1.12699224763999035261E-8;
    fd = fd*u+3.60140029589371370404E-11;
    fd = fd*u+5.88754533621578410010E-14;
    fd = fd*u+4.52001434074129701496E-17;
    fd = fd*u+1.25443237090011264384E-20;
    gn = 5.04442073643383265887E-1;
    gn = gn*u+1.97102833525523411709E-1;
    gn = gn*u+1.87648584092575249293E-2;
    gn = gn*u+6.84079380915393090172E-4;
    gn = gn*u+1.15138826111884280931E-5;
    gn = gn*u+9.82852443688422223854E-8;
    gn = gn*u+4.45344415861750144738E-10;
    gn = gn*u+1.08268041139020870318E-12;
    gn = gn*u+1.37555460633261799868E-15;
    gn = gn*u+8.36354435630677421531E-19;
    gn = gn*u+1.86958710162783235106E-22;
    gd = 1.00000000000000000000E0;
    gd = gd*u+1.47495759925128324529E0;
    gd = gd*u+3.37748989120019970451E-1;
    gd = gd*u+2.53603741420338795122E-2;
    gd = gd*u+8.14679107184306179049E-4;
    gd = gd*u+1.27545075667729118702E-5;
    gd = gd*u+1.04314589657571990585E-7;
    gd = gd*u+4.60680728146520428211E-10;
    gd = gd*u+1.10273215066240270757E-12;
    gd = gd*u+1.38796531259578871258E-15;
    gd = gd*u+8.39158816283118707363E-19;
    gd = gd*u+1.86958710162783236342E-22;
    f = 1-u*fn/fd;
    g = t*gn/gd;
    t = mpio2*x2;
    cc = cos(t);
    ss = sin(t);
    t = mpi*x;
    *c = 0.5+(f*ss-g*cc)/t;
    *s = 0.5-(f*cc+g*ss)/t;
//    c = c*ap::sign(xxa);
//    s = s*ap::sign(xxa);
    *c = *c*xxa/fabs(xxa);
    *s = *s*xxa/fabs(xxa);
}


void WaveH10New(double b,double u,double *ReF0,double *ImF0)
{
double y,Dy,re,im,pre,pim;
double t1,t2,q,g,v1,v2,gp,gm;
double ct1,ct2,cv1,cv2,st1,st2,sv1,sv2;

q=sqrt(2.0*b/PI);
gp=(u+PI/2.0)/2.0/b;
gm=(u-PI/2.0)/2.0/b;

//------Пределы интегрирования--------//
t1=-q*(1.0+gp);
t2=q*(1.0-gp);
v1=-q*(1.0+gm);
v2=q*(1.0-gm);
//------------------------------------//
//--------Интегралы Френеля-----------//
fresnelintegral(t1,&ct1,&st1);
fresnelintegral(t2,&ct2,&st2);
fresnelintegral(v1,&cv1,&sv1);
fresnelintegral(v2,&cv2,&sv2);
//----Реальная и Мнимая  часть H10----//
re=cos(gp*gp*b)*(ct2-ct1)+sin(gp*gp*b)*(st2-st1)+cos(gm*gm*b)*(cv2-cv1)+sin(gm*gm*b)*(sv2-sv1);
re*=0.5/q;
//re*=DR/q/4.0;
im=sin(gp*gp*b)*(ct2-ct1)-cos(gp*gp*b)*(st2-st1)+sin(gm*gm*b)*(cv2-cv1)-cos(gm*gm*b)*(sv2-sv1);
im*=0.5/q;
//im*=DR/q/4.0;
//------------------------------------//
*ReF0=re;
*ImF0=im;
}

void WaveH2N0New(double b,double u,int n,double *ReFN,double *ImFN)
{
double y,Dy,re,im,pre,pim;
double t1,t2,q,g,v1,v2,gp,gm;
double ct1,ct2,cv1,cv2,st1,st2,sv1,sv2;

q=sqrt(2.0*b/PI);
gp=(u+PI*(double)(n))/2.0/b;
gm=(u-PI*(double)(n))/2.0/b;

//------Пределы интегрирования--------//
t1=-q*(1.0+gp);
t2=q*(1.0-gp);
v1=-q*(1.0+gm);
v2=q*(1.0-gm);
//------------------------------------//
//--------Интегралы Френеля-----------//
fresnelintegral(t1,&ct1,&st1);
fresnelintegral(t2,&ct2,&st2);
fresnelintegral(v1,&cv1,&sv1);
fresnelintegral(v2,&cv2,&sv2);
//----Реальная и Мнимая  часть H10----//
im=-(cos(gp*gp*b)*(ct2-ct1)+sin(gp*gp*b)*(st2-st1)-cos(gm*gm*b)*(cv2-cv1)-sin(gm*gm*b)*(sv2-sv1));
im*=0.5/q;
//im*=DR/q/4.0;
re=sin(gp*gp*b)*(ct2-ct1)-cos(gp*gp*b)*(st2-st1)-sin(gm*gm*b)*(cv2-cv1)+cos(gm*gm*b)*(sv2-sv1);
re*=0.5/q;
//im*=DR/q/4.0;
//------------------------------------//
*ReFN=re;
*ImFN=im;
}

void Initialization_ModelKvadratDNR_Meredian(double freq,double *Amp0Max,double *AmpUMax){
	int i;
	double dgam,gam,gam0,Gam1,Gam2,lbd;
	double ReF0,ImF0,ReF2,ImF2,ReF4,ImF4,ReF6,ImF6,ReF8,ImF8;
	double ReU,ImU,ReD,ImD,AmpU,AmpD,Amp0,AmpResivU,AmpResivD;
	double a6,f6,a8,f8,faza,faza0,u,u90,k;
	double a2,f2,a4,f4,b;
	double DR=12.2;  //Ширина рупора в метрах
	double LR=21.27; //Эффективная длинна рупора в метрах

	lbd=VC/1.0E6/freq;//Длина волны в метрах
	b=60.0*rad;
	a2=0.812964006;
	f2=14.6*rad;
	a4=0.3;
	f4=250.0*rad;
	a6=0.0;
	f6=-170.0*rad;
	a8=0.0;
	f8=36.0*rad;

	gam0=10.0*rad;
	dgam=0.1*rad;
	k=2.0*PI/lbd;//Волновое число
	u90=DR*k/2.0;

	*AmpUMax=0.0;
	*Amp0Max=0.0;

	Gam1=-gam0;
	Gam2= gam0;

	for(gam=Gam1;gam<=Gam2;gam+=dgam){
		u=u90*sin(gam);
	//---------------------------------------------------------//
		WaveH10New(b,u,&ReF0,&ImF0);
		WaveH2N0New(b,u,1,&ReF2,&ImF2);
		WaveH2N0New(b,u,2,&ReF4,&ImF4);
		WaveH2N0New(b,u,3,&ReF6,&ImF6);
		WaveH2N0New(b,u,4,&ReF8,&ImF8);
	//---------------------------------------------------------//

		ReU=ReF0+a2*(ReF2*cos(f2)-ImF2*sin(f2));
		ImU=ImF0+a2*(ReF2*sin(f2)+ImF2*cos(f2));
		ReU-=a4*(ReF4*cos(f4)-ImF4*sin(f4));
		ImU-=a4*(ReF4*sin(f4)+ImF4*cos(f4));
		ReU+=a6*(ReF6*cos(f6)-ImF6*sin(f6));
		ImU+=a6*(ReF6*sin(f6)+ImF6*cos(f6));
		ReU-=a8*(ReF8*cos(f8)-ImF8*sin(f8));
		ImU-=a8*(ReF8*sin(f8)+ImF8*cos(f8));
		
		ReD=ReF0-a2*(ReF2*cos(f2)-ImF2*sin(f2));
		ImD=ImF0-a2*(ReF2*sin(f2)+ImF2*cos(f2));
		ReD+=a4*(ReF4*cos(f4)-ImF4*sin(f4));
		ImD+=a4*(ReF4*sin(f4)+ImF4*cos(f4));
		ReD-=a6*(ReF6*cos(f6)-ImF6*sin(f6));
		ImD-=a6*(ReF6*sin(f6)+ImF6*cos(f6));
		ReD+=a8*(ReF8*cos(f8)-ImF8*sin(f8));
		ImD+=a8*(ReF8*sin(f8)+ImF8*cos(f8));
		
		AmpU=sqrt(ReU*ReU+ImU*ImU);
		AmpD=sqrt(ReD*ReD+ImD*ImD);
		Amp0=2.0*sqrt(ReF0*ReF0+ImF0*ImF0);

		if(*AmpUMax<AmpU)
			*AmpUMax=AmpU; 
		if(*Amp0Max<Amp0)
			*Amp0Max=Amp0;
	}
}

void ModelDNR_Meredian(double freq,double gam_user,double *AMP0,double *AMPU, double *AMPD, double *Faza){
int i;
double dgam,gam,gam0,Gam1,Gam2,lbd;
double ReF0,ImF0,ReF2,ImF2,ReF4,ImF4,ReF6,ImF6,ReF8,ImF8;
double ReU,ImU,ReD,ImD,AmpU,AmpD,Amp0;
double a6,f6,a8,f8,faza,faza0,u,u90,k;
double fazaadd,fold,FAZA=0.0,AmpMax,AmpUMax,Amp0Max,gammax;
double a2,f2,a4,f4,b; 
double DR=12.2;  //Ширина рупора в метрах
double LR=21.27; //Эффективная длинна рупора в метрах

lbd=VC/1.0E6/freq;//Длина волны в метрах
b=60.0*rad; 
a2=0.812964006;
f2=14.6*rad;
a4=0.3;
f4=250.0*rad; 
a6=0.0;
f6=-170.0*rad;
a8=0.0;
f8=36.0*rad;

k=2.0*PI/lbd;//Волновое число
u90=DR*k/2.0;

u=u90*sin(gam_user);
//---------------------------------------------------------//
WaveH10New(b,u,&ReF0,&ImF0);
WaveH2N0New(b,u,1,&ReF2,&ImF2);
WaveH2N0New(b,u,2,&ReF4,&ImF4);
WaveH2N0New(b,u,3,&ReF6,&ImF6);
WaveH2N0New(b,u,4,&ReF8,&ImF8);
//---------------------------------------------------------//

ReU=ReF0+a2*(ReF2*cos(f2)-ImF2*sin(f2));
ImU=ImF0+a2*(ReF2*sin(f2)+ImF2*cos(f2));
ReU-=a4*(ReF4*cos(f4)-ImF4*sin(f4));
ImU-=a4*(ReF4*sin(f4)+ImF4*cos(f4));
ReU+=a6*(ReF6*cos(f6)-ImF6*sin(f6));
ImU+=a6*(ReF6*sin(f6)+ImF6*cos(f6));
ReU-=a8*(ReF8*cos(f8)-ImF8*sin(f8));
ImU-=a8*(ReF8*sin(f8)+ImF8*cos(f8));
		
ReD=ReF0-a2*(ReF2*cos(f2)-ImF2*sin(f2));
ImD=ImF0-a2*(ReF2*sin(f2)+ImF2*cos(f2));
ReD+=a4*(ReF4*cos(f4)-ImF4*sin(f4));
ImD+=a4*(ReF4*sin(f4)+ImF4*cos(f4));
ReD-=a6*(ReF6*cos(f6)-ImF6*sin(f6));
ImD-=a6*(ReF6*sin(f6)+ImF6*cos(f6));
ReD+=a8*(ReF8*cos(f8)-ImF8*sin(f8));
ImD+=a8*(ReF8*sin(f8)+ImF8*cos(f8));
		
AmpU=sqrt(ReU*ReU+ImU*ImU);
AmpD=sqrt(ReD*ReD+ImD*ImD);
Amp0=2.0*sqrt(ReF0*ReF0+ImF0*ImF0);

Amp0Max=0.0;
AmpUMax=0.0;
for(i=0;i<=M0U;i++){
	Amp0Max+=KAmp0[i]*pow(freq,(double)(M0U-i));
	AmpUMax+=KAmpU[i]*pow(freq,(double)(M0U-i));
}

// AmpUMax == AmpDMax
*Faza=-atan((ReU*ImD-ReD*ImU)/(ReU*ReD+ImU*ImD));
*AMP0=Amp0/Amp0Max; // Амплитуда для синфазной работы полурупоров (передача либо прием)
*AMPU=AmpU/AmpUMax; // Амплитуда для передачи либо приема верхним полурупором
*AMPD=AmpD/AmpUMax; // Амплитуда для передачи либо приема нижним полурупором

// Примеры:
// Передача - оба полурупора, прием - верхний
// либо передача - верхний, прием - оба
// AmpU*Amp0/AmpUMax/Amp0Max = *AMP0 * *AMPU

// Передача - оба полурупора, прием - нижний
// либо передача - нижний, прием - оба
// AmpD*Amp0/AmpUMax/Amp0Max = *AMP0 * *AMPD
}


double AzimMaxDNR(double freqkHz){//Возвращаем угол в радианах
	double ep0;
	ep0=185634.9983-4.877867876*freqkHz+4.803667396E-005*freqkHz*freqkHz-2.102600271E-010*freqkHz*freqkHz*freqkHz+3.453540782E-016*freqkHz*freqkHz*freqkHz*freqkHz;
	ep0+=DEP;
	ep0*=rad;
	return ep0;
}

double AzimMaxDNR_model(double freqkHz){//Возвращаем угол в радианах
	double ep0;
	int N=283; //Число щелей в структуре
	double HRup=246.0; //Длинна рупора в метрах
	double a=0.136;//Ширина канавки в замедляющей структуре в м
	double b=0.014;//Ширна ребра гребенки в м
	double h=0.384;//Высота гребенки в м
	double d=HRup/(double)(N-1); //Расстояние между щелями, примерно, 87см
	double lambda = VC/freqkHz/1000.0;
	double k=2.0*PI/lambda;//*freq*1.0E6/VC;
	double freq = freqkHz/1000.0;

	double kstr=0.0;
	for(int i=0;i<=Mstr;i++){
		kstr+=Kstr[i]*pow(freq, (double)(Mstr-i));
	}
	double g=kstr*sqrt(1.0+pow(a*tan(k*h)/(a+b),2.0));
	ep0 = asin(g-lambda/d);

	return ep0;
}

double DNR_Azimut_Power(double freq,double ep){
	int i;
	double d,norm,g,ep0,kstr;
	double Power,x,y,psi;
	int N=283; //Число щелей в структуре
	double HRup=246.0; //Длинна рупора в метрах
	double a=0.136;//Ширина канавки в замедляющей структуре в м
	double b=0.014;//Ширна ребра гребенки в м
	double h=0.384;//Высота гребенки в м
	double lambda = VC/freq/1.0E6;
	double k=2.0*PI/lambda;//*freq*1.0E6/VC;

	kstr=0.0;
	for(i=0;i<=Mstr;i++){
		kstr+=Kstr[i]*pow(freq,(double)(Mstr-i));
	}
	g=kstr*sqrt(1.0+pow(a*tan(k*h)/(a+b),2.0));
	ep0=AzimMaxDNR_model(freq*1000.0);

	d=HRup/(double)(N-1); //Расстояние между щелями, примерно, 87см

	norm=(cosh(alfa*d)-1.0)/(cosh(alfa*HRup)-1.0);
	psi=k*d*(g-sin(ep));
	y=cosh(alfa*HRup)-cos((double)(N)*psi);
	x=cosh(alfa*d)-cos(psi);
	Power=norm*y/x;

	return Power;
}

void InitializationParamDNR(void){
    double freq_low = 153.0;
    double freq_upp = 163.0;
	int i=0,I,j;
	double freq,gam;
	double AMP0_Max,AMPU_Max;
	double X[1000],Y[1000],Y2[1000];
	double AMP0,AMPU,AMPResivU,AMPD,AMPResivD,Faza;

	double freqkHz,a,b,h,lb,k,g,kstr,ep,Pep;
	int	N=283; //Число щелей в структуре
	double HRup=246.0; //Длинна рупора в метрах
	double d=HRup/(double)(N-1); //Расстояние между щелями, примерно, 87см
	double delt=0.25;//Уровень мощности в конце антенны

	for(freq=freq_low;freq<=freq_upp;freq+=0.1){
		X[i]=freq;
		Initialization_ModelKvadratDNR_Meredian(freq,&AMP0_Max,&AMPU_Max);
		Y[i]=AMP0_Max;
		Y2[i]=AMPU_Max;
		i++;
	}
	I=i-1;
	Calculate(M0U,I,X,Y,KAmp0);
	Calculate(M0U,I,X,Y2,KAmpU);

	if(delt!=0.0)
		alfa=log(delt)/HRup;//Показатель в экспоненте (скорость падения поля)
	else
		alfa=0.0;

	a=0.136;//Ширина канавки в замедляющей структуре в м
	b=0.014;//Ширна ребра гребенки в м
	h=0.384;//Высота гребенки в м

	i=0;
	for(freq=freq_low;freq<=freq_upp;freq+=0.1){
		freqkHz=freq*1000.0;
		X[i]=freq;

		lb=VC/1.0E6/freq;//Длина волны в воздухе (м)
		k=2.0*PI/lb; //Волновое число
		g=sqrt(1.0+pow(a*tan(k*h)/(a+b),2.0));//Коэффициент замедления без учета конечного размера структуры
		ep=AzimMaxDNR(freqkHz);
		Y[i]=(sin(ep)+lb/d)/g;//Коэффициент учета конечного размера структуры

		i++;
	}
	i--;
	Calculate(Mstr,i,X,Y,Kstr);

}


double Cardioid(double ep, double gamma){
    return 0.5 * (1 + cos(ep) * cos(gamma));
}


double DnrInMax(double freq, int dnr_type){
    double ep_max = AzimMaxDNR_model(freq * 1000.);
    double Pep, Pgam;
    double Amp0, AMPU, AMPD, Faza, Amp;

    const double gamma = 0.;

    Pep = DNR_Azimut_Power(freq, ep_max);
    ModelDNR_Meredian(freq, gamma, &Amp0, &AMPU, &AMPD, &Faza);

    if (dnr_type == 0)
        Pgam = Amp0 * Amp0;
    else if (dnr_type == 1)
        Pgam = AMPU * AMPU;
    else if (dnr_type == 2)
        Pgam = AMPD * AMPD;
    else
        return -1;
    return Pep * Pgam * pow(Cardioid(ep_max, gamma), 2);
}


double CalcDNR(double freq, double ep, double gam, int dnr_type){
    /*
    Типы диаграммы (dnr_type):
        0 - Оба полурупора
        1 - Верхний полурупор
        2 - Нижний полурупор
		3 - Фаза
    */
	double Pep, Pgam, P;
	double ep_max;
	double theta;
	double Amp0, AMPU, AMPD, Faza, Amp;

	Pep=DNR_Azimut_Power(freq, ep);//ДН в азимутальном направлении по мощности
	ModelDNR_Meredian(freq, gam, &Amp0, &AMPU, &AMPD, &Faza);

	if (dnr_type == 0)
	    Amp = Amp0;
	else if (dnr_type == 1)
	    Amp = AMPU;
	else if (dnr_type == 2)
	    Amp = AMPD;
	else if (dnr_type == 3)
		return Faza;
	else
	    return -1;

	Pgam = Amp * Amp;

	// No normalization after cardioid. Overall Pattern = (F_elem * F_array)^2
	P = Pgam * Pep * pow(Cardioid(ep, gam), 2);
	return P;
}

// Моностатическая диаграмма по мощности (прием и передача)
double MonostaticDNR(double freq, double ep, double gam, int tr_type, int rc_type){
	double Pep, P, Pgam_tr, Pgam_rc, Ptr, Prc;
	double Amp0, AMPU, AMPD, Faza, Amp;
	double theta;
	double *Amp_tr, *Amp_rc;

	Pep=DNR_Azimut_Power(freq, ep);//ДН в азимутальном направлении по мощности
	ModelDNR_Meredian(freq, gam, &Amp0, &AMPU, &AMPD, &Faza);

	void choose_type(int type, double** ampl){
        if (type == 0)
	    	*ampl = &Amp0;
	    else if (type == 1)
	        *ampl = &AMPU;
	    else if (type == 2)
	        *ampl = &AMPD;
	    else
	        *ampl = NULL;
	}

	choose_type(tr_type, &Amp_tr);
	choose_type(rc_type, &Amp_rc);

	Pgam_tr = *Amp_tr * *Amp_tr;
	Pgam_rc = *Amp_rc * * Amp_rc;

    // No normalization after cardioid. Overall Pattern = (F_elem * F_array)^2
	Ptr = Pgam_tr * Pep * pow(Cardioid(ep, gam), 2);
	Prc = Pgam_rc * Pep * pow(Cardioid(ep, gam), 2);
	
	P = Ptr * Prc;
	return P;
}

int CalcDNR_arrays(double freq, double* ep, int ep_size, double* gam, int gam_size,
					int dnr_type, double* result){
	double Pep, Pgam;
	double Amp0, AMPU, AMPD, Faza;
	double theta;
	double *Amp;
	int curr_iter;

	if (dnr_type == 0)
	    Amp = &Amp0;
	else if (dnr_type == 1)
	    Amp = &AMPU;
	else if (dnr_type == 2)
	    Amp = &AMPD;
	else
	    return -1;

    int i, j;
	for (i=0; i<ep_size; i++){
	    for (j=0; j<gam_size; j++){
	        //ДН в азимутальном направлении по мощности
	        curr_iter = i * gam_size + j;
            Pep=DNR_Azimut_Power(freq, ep[curr_iter]);
            ModelDNR_Meredian(freq, gam[curr_iter], &Amp0, &AMPU, &AMPD, &Faza);
            Pgam = *Amp * *Amp;

            // No normalization after cardioid. Overall Pattern = (F_elem * F_array)^2
            result[curr_iter] = Pgam * Pep * pow(Cardioid(ep[curr_iter], gam[curr_iter]), 2);
	    }
	}
}


void MonostaticDNR_arrays(double freq, double* ep, int ep_size, double* gam, int gam_size,
						  int tr_type, int rc_type, double* result){
	double Pep, Pgam_tr, Pgam_rc, Ptr, Prc;
	double Amp0, AMPU, AMPD, Faza;
	double *Amp_tr, *Amp_rc;
	double theta;
	int curr_iter = 0;

	void choose_type(int type, double** ampl){
        if (type == 0)
	    	*ampl = &Amp0;
	    else if (type == 1)
	        *ampl = &AMPU;
	    else if (type == 2)
	        *ampl = &AMPD;
	    else
	        *ampl = NULL;
	}

	choose_type(tr_type, &Amp_tr);
	choose_type(rc_type, &Amp_rc);
    int i, j;
	for (i=0; i<ep_size; i++){
	    for (j=0; j<gam_size; j++){
	        curr_iter = i * gam_size + j;
	        //ДН в азимутальном направлении по мощности
            Pep=DNR_Azimut_Power(freq, ep[curr_iter]);
            ModelDNR_Meredian(freq, gam[curr_iter], &Amp0, &AMPU, &AMPD, &Faza);
            Pgam_tr = *Amp_tr * *Amp_tr;
            Pgam_rc = *Amp_rc * *Amp_rc;

            // No normalization after cardioid. Overall Pattern = (F_elem * F_array)^2
            Ptr = Pgam_tr * Pep * pow(Cardioid(ep[curr_iter], gam[curr_iter]), 2);
            Prc = Pgam_rc * Pep * pow(Cardioid(ep[curr_iter], gam[curr_iter]), 2);
            result[curr_iter] = Ptr * Prc;
	    }
	}
}


void CalcDNR_In_Star(void){
	FILE*faz;
	int i;
	int year,month,day;
	double t; //Время в долях суток
	double freq=155.8;
	double gam,Amp0,AMPU,AMPResivU,AMPD,AMPResivD,Faza,Pmin;
	double ep,AmpEp,P;
	double el,az,x,R;
	double Ua=7.0*rad;
	double Ub=10.0*rad;
	double LG0=103.257*rad;
	double FI0=52.881389*rad;
	double H0=0.502;
	double LG,FI,H; 
	double Ra,Dec;

	year=2015;
	month=6;
	day=8;
	t=20.738889*3600.0;
	Ra=(20.0+0.0/60.0+0.89/3600.0)*15.0*rad;
	Dec=(40.0+45.0/60.0+50.03/3600.0)*rad;
	Star_Radar(year,month,day,t,LG0,FI0,Ra,Dec,&el,&az,&ep,&gam);//Переход из звездной в антенную СК
	P=CalcDNR(freq,ep,gam,0);
	
	faz=fopen("DNR_in_Star155_8.dat","w");
	fprintf(faz,"Ra	Dec	P\n");
	for(Ra=0.0*rad;Ra<360*rad;Ra+=0.2*rad){
		printf("Ra=%lf\r",Ra/rad);
		for(Dec=0.0*rad;Dec<90*rad;Dec+=0.1*rad){
			Star_Radar(year,month,day,t,LG0,FI0,Ra,Dec,&el,&az,&ep,&gam);//Переход из звездной в антенную СК
			P=CalcDNR(freq,ep,gam, 0);
			fprintf(faz,"%lf	%lf	%lf\n",Ra/rad,Dec/rad,10.0*log10(P));
		}
	}
	fclose(faz);

}

//int main(void){
//    InitializationParamDNR();
//    int npoints = 100;
//
//    printf("Create ep gam\n");
//
//	double *ep = (double*) malloc(npoints*npoints*sizeof(double));
//	double *gam = (double*) malloc(npoints*npoints*sizeof(double));
//
//	printf("Init ep gam\n");
//	int i,j;
//	double az, el;
//	for (i=0; i<npoints; i++){
//	    az = 0 + 2*PI*((double)i/(double)npoints);
//	    for (j=0; j<npoints; j++){
//	        el = PI/2 - PI/2*((double)j/(double)npoints);
//	        Topoc_to_Ant(el, az, &gam[i*npoints + j], &ep[i*npoints + j]);
//	    }
//	}
//
//	printf("Create res\n");
//	double* res = (double*) malloc(npoints*npoints*sizeof(double));
//	printf("Calc res\n");
//	CalcDNR_arrays(154., ep, npoints, gam, npoints, 0, res);
//
//    printf("Saving\n");
//	FILE *file = fopen("pat_log.dat", "w");
//	for (i=0; i<npoints; i++){
//	    for (j=0; j<npoints; j++){
//	        fprintf(file, "ep %f gam %f pat %e\n", ep[i*npoints+j],
//	                gam[i*npoints+j], res[i*npoints+j]);
//	    }
//	}
//
//	fclose(file);
//	printf("End\n");
//	return 1;
//}