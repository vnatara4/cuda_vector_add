#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
	FILE *fp;
	char tmp;
	int a[10][10], i = 0, j = 0, l, k;
	fp = fopen(argv[1], "r");

	if (fp != NULL) {
		while (!feof(fp)) {
			tmp = fgetc(fp);
			if(tmp == '\n') {
				i++;
				j = 0;
			}
			fseek(fp, 2, SEEK_CUR);
			fscanf(fp, "%d", &a[i][j]);
			printf("a[%d][%d] : %d\n", i, j, a[i][j]);
			j++;
		}
	}
	for(k = 0; k < i; k++) {
		for(l = 0; l < j; l++){
			printf("%d ", a[k][l]);		
		}
		printf("\n");
	}
}
