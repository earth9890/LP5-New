#include<bits/stdc++.h>
#include<omp.h>

using namespace std;

void printa(int arr[])
{
	for(int i = 0; i < 10; i++)
	{
		cout << arr[i] << " ";
	}
	cout << endl;
}

void bubble(int arr[], int n)
{
	for(int i = 0; i < n-1; i++)
	{
		for(int j = 0; j < n-i-1; j++)
		{
			if(arr[j] > arr[j+1])
				swap(arr[j], arr[j+1]);
			cout << "itr " << i+1 << "pass" << j+1 << ": ";
			printa(arr);
		}
			
	}
}

void parallelb(int arr[], int n)
{
	for(int i = 0; i < n; i++)
	{
		#pragma omp parallel for
		for(int j = 1; j < n; j += 2)
		{
			if(arr[j] < arr[j-1])
				swap(arr[j], arr[j-1]);
		}
		#pragma omp barrier
		#pragma omp parallel for
		for(int j = 2; j < n; j += 2)
		{
			if(arr[j] < arr[j-1])
				swap(arr[j], arr[j-1]);
		}
	}
	//printa(arr);
}

void merge(int arr[], int low, int mid, int high)
{
	int n1 = mid-low+1;
	int n2 = high-mid;
	int left[n1];
	int right[n2];
	
	for(int i = 0; i < n1; i++)
		left[i] = arr[low+i];
	for(int i = 0; i < n2; i++)
		right[i] = arr[mid+1+i];
	
	int i = 0, j = 0;
	int k = low;
	
	while(i < n1 && j < n2)
	{
		if(left[i] <= right[j])
		{
			arr[k] = left[i];
			i++;
			k++;
		}
		else
		{
			arr[k] = right[j];
			j++;
			k++;
		}
	}
	while(i < n1)
	{
		arr[k] = left[i];
		i++;
		k++;	
	}
	while(j < n2)
	{
		arr[k] = right[j];
		j++;
		k++;	
	}
	
	cout << "Array after merge: ";
    for (int idx = low; idx <= high; idx++) {
        cout << arr[idx] << " ";
    }
    cout << endl;
}

void merges(int arr[], int low, int high)
{
	int mid = (low+high)/2;
	if(low < high)
	{
		merges(arr, low, mid);
		merges(arr, mid+1, high);
		merge(arr, low, mid, high);
	}
	//printa(arr);
}

void pms(int arr[], int low, int high)
{
	int mid = (low+high)/2;
	if(low < high)
	{
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			pms(arr, low, mid);
		}
		#pragma omp section
		{
			pms(arr, mid+1, high);
		}
	}
		merge(arr, low, mid, high);
	}
	//printa(arr);
}

	
	

int main()
{
	int n = 10;
	int arr[n];
	
	for(int i = 0; i < n; i++)
	{
		int x = rand() % 10;
		arr[i] = x;
	}
	
	
	auto st = omp_get_wtime();
	bubble(arr, n);
	auto ed = omp_get_wtime();
	cout << (ed-st)*1e6 << endl;
	
	st = omp_get_wtime();
	parallelb(arr, n);
	ed = omp_get_wtime();
	cout << (ed-st)*1e6 << endl;
	
	st = omp_get_wtime();
	merges(arr, 0, n-1);
	ed = omp_get_wtime();
	cout << (ed-st)*1e6 << endl;
	
	st = omp_get_wtime();
	pms(arr, 0, n-1);
	ed = omp_get_wtime();
	cout << (ed-st)*1e6 << endl;
	
	return 0;
}


	





















