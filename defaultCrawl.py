import scrapy 
import re
from pymongo import MongoClient

class defaultSpider(scrapy.Spider):
    """docstring for CourseWebSpider"""
    name = "defaultCrawl"
    start_urls = ["http://comp4332.com/trial"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        client = MongoClient("mongodb://localhost:27017")
        db = client["university"]


    def parse(self, response):
        listOfLink = response.xpath("//a[@href]/@href").extract()
        for link in listOfLink:
            yield response.follow(link, callback = self.parse_deptPages)

    # crawl each department (68) inside time slot page
    def parse_deptPages(self, response):
        listOfDept = response.xpath("//a[starts-with(@href, 'subjects')]/@href").extract()
        for dept in listOfDept:
            yield response.follow(dept, callback = self.parse_insert)

    # crawl each course in each department.timeslot combination (4*68)
    # there will be two sets of data, courseInfo and sectionInfo
    # two databases would be connected by cid (course ID, e.g. ACCT 1010, LIFS 1020)
    def parse_insert(self, response):
        # get course information of each department
        # cid = course ID, ctitle = course title, credit = number of units carried, description = description in COURSE INFO popup
        for record in response.xpath('//div[@class="course"]'):
            cid = record.xpath('.//a/@name').extract_first()
            titleAndcredit = record.xpath('.//h2/text()').extract_first()

            description = (record.xpath('.//tr[th="DESCRIPTION"]/td/text()').extract_first()).encode(response.encoding)
            string = re.split(r'[ ()]', titleAndcredit)
            ctitle = " ".join(string[3:-4:1])
            credit = string[-3]

            db.course.insert_one({
                "cid": cid,
                "ctitle": ctitle,
                "credit": credit,
                "describe": description
                })
            db.course.pop('_id', None)

        for record in response.xpath('//tr[starts-with(@class, "newsect")]'):
            timeslot_string = record.xpath('./ancestor::html/head/title/text()').extract_first()
            ts = timeslot_string.split(" ")[6] + "T" + timeslot_string.split(" ")[7] + ":00"
            dateTimeVariable = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")

            cid = record.xpath('./ancestor::div[@class="course"]//a/@name').extract_first()
            section = record.xpath('./td[1]/text()').extract_first()

            # there are some sections with multiple class time (DateTime), venue (room), instructor (instructor)
            DateTime = record.xpath('./td[2]/text()').extract_first()
            room = record.xpath('./td[3]/text()').extract_first()
            instructor = record.xpath('./td[4]/text()').extract_first()
            if record.xpath('./td[5]/span/text()').extract_first() == None:
                quota = record.xpath('./td[5]/text()').extract_first()
            else:
                quota = record.xpath('./td[5]/span/text()').extract_first()
            enrol = record.xpath('./td[6]/text()').extract_first()
            if record.xpath('./td[7]/strong/text()').extract_first() == None:
                avail = record.xpath('./td[7]/text()').extract_first()
            else:
                avail = record.xpath('./td[7]/strong/text()').extract_first()
            if record.xpath('./td[8]/strong/text()').extract_first() == None:
                wait = record.xpath('./td[8]/text()').extract_first()
            else:
                wait = record.xpath('./td[8]/strong/text()').extract_first()
            remarks = record.xpath('./td[9]//div[@class="popupdetail"]/text()').extract_first()

            db.secList.insert_one({
                "ts": dateTimeVariable,
                "cid": cid,
                "section": section,
                "DT": DateTime,
                "room": room,
                "instructor": instructor,
                "quota": quota,
                "enrol": enrol,
                "avail": avail,
                "wait": wait,
                "remarks": remarks
                })
            db.secList.pop('_id', None)
                
