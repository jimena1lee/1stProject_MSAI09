# 전처리 결과
1. 스케일링 완료 데이터
   * 성남 : seongnam_scaled.csv
   * 광명 : gwangmyung_scaled.csv
   * feature : 도로안전표지,도로적색표면,무단횡단방지펜스,무인교통단속카메라,보호구역표지판,생활안전CCTV,신호등,옐로카펫,횡단보도,어린이 비율(%),발생건수
     
2. 스케일링된 데이터 + 시설물,위도,경도 포함
   * 성남 : seongnam_scaled_with_info.csv
   * 광명 : gwangmyung_scaled_with_info.csv
   * feature : 시설물명,위도,경도,도로안전표지,도로적색표면,무단횡단방지펜스,무인교통단속카메라,보호구역표지판,생활안전CCTV,신호등,옐로카펫,횡단보도,어린이 비율(%),발생건수
     
2. 스케일링 전후 분포 비교
   * 성남 : facility_feature_summary_sn.csv
   * 광명 : facility_feature_summary_gm.csv
   * feature : ,skewness_전,skewness_후,mean_후,std_후,변환방식,skewness_개선
