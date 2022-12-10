#include<bits/stdc++.h>
using namespace std;
double w[99],u,r[99],W[99],v;
int A,I,a[9999][99];
void D(int x){
	if(x==3){
		//acc
		A=0;
		for(int i=1;i<=1998;i++){
			for(int j=0;j<=9;j++)
				r[j]=0;
			for(int j=1;j<=3;j++)
				r[a[i][j]]+=w[j];
			u=0;
			for(int j=0;j<=9;j++)
				if(r[j]>u)
					u=r[j],I=j;
			A+=(I==a[i][0]);
		}
		
		//if acc > max_acc
		if(A>v||A==v&&w[1]+w[2]<W[1]+W[2]){
			v=A;
			for(int i=1;i<=2;i++)
				W[i]=w[i];
		}
		return;
	}
	for(int i=1;i<=10;i++)
		w[x]=i,D(x+1);
}
main(){
	freopen("a.out","r",stdin);
	for(int i=1;i<=1998;i++)
		for(int j=0;j<=1;j++)
			cin>>a[i][j];
	for(int i=1;i<=1998;i++)
		cin>>a[i][2];
	D(1);
	cout<<1.0*v/1998<<' '<<W[1]<<' '<<W[2];//<<' '<<W[3];
}
/*0.9474
0.9479
0.9434
0.953453 2 2 1

8 7 9 2 1 3
3
xgb
rf
knn*/
