int main(int argc, char *argv[]) {
	File *fp;
	char tmp1, tmp2;
	int a[10][10]. b[10][10], i = 0, j = 0, status = 0;
	fp = fopen(argv[1], "r");
	while(!feof(fp)) {
		tmp1 = fgetc(fp);
		if(fp == ' ')
			continue;
		if(fp != '\n') {
			if(status == 0)
				a[i][j] = atoi(tmp1);
			else
				b[i][j] = atoi(tmp2);
			j++;
		} else {
			tmp2 = fgetc(fp);
			i++;
			if(tmp2 != '\n') {
				if(status == 0)
					a[i][j] = atoi(tmp2);										
				else
					b[i][j] = atoi(tmp2);
			} else {
				status = 1;
				i = 0;
				j = 0;
			} 
		} 
			
		
	}		

}
