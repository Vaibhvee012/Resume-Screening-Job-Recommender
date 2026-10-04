"""Dictionary-based skill extraction with alias normalisation.
'node js', 'nodejs', 'node.js' all normalise to the canonical skill 'Node.js'."""
import re

# canonical name -> list of aliases (lowercase). Canonical name is also matched.
SKILL_ALIASES = {
    # programming languages
    "Python": ["python"], "Java": ["java", "core java", "j2ee", "jdk"], "JavaScript": ["javascript", "js", "ecmascript", "es6"],
    "TypeScript": ["typescript"], "C": [], "C++": ["c++", "cpp"], "C#": ["c#", "c sharp", "csharp"],
    "PHP": ["php"], "Ruby": ["ruby", "ruby on rails", "rails"], "Go": ["golang"], "R": [], "Scala": ["scala"],
    "Kotlin": ["kotlin"], "Swift": ["swift"], "Perl": ["perl"], "MATLAB": ["matlab"], "Shell Scripting": ["shell scripting", "bash", "shell script", "unix shell"],
    # web
    "HTML": ["html", "html5"], "CSS": ["css", "css3"], "React": ["react", "reactjs", "react.js", "react js"],
    "Angular": ["angular", "angularjs", "angular.js"], "Vue.js": ["vue", "vuejs", "vue.js"], "Node.js": ["node.js", "nodejs", "node js"],
    "jQuery": ["jquery"], "Bootstrap": ["bootstrap"], "Django": ["django"], "Flask": ["flask"], "Spring": ["spring", "spring boot", "spring mvc"],
    "ASP.NET": ["asp.net", "asp net", "asp dot net", "asp.net mvc"], ".NET": [".net", "dotnet", "dot net"], "WordPress": ["wordpress"],
    "REST API": ["rest api", "restful", "rest apis", "web services", "restful api"], "AJAX": ["ajax"], "XML": ["xml"], "JSON": ["json"],
    "Magento": ["magento"], "Drupal": ["drupal"], "Laravel": ["laravel"], "CodeIgniter": ["codeigniter"], "Photoshop": ["photoshop"],
    # mobile
    "Android": ["android"], "iOS": ["ios", "iphone"], "React Native": ["react native"], "Flutter": ["flutter"],
    # databases
    "SQL": ["sql", "t-sql", "pl/sql", "plsql", "pl sql"], "MySQL": ["mysql"], "PostgreSQL": ["postgresql", "postgres"],
    "Oracle": ["oracle", "oracle dba"], "SQL Server": ["sql server", "mssql", "ms sql", "ms sql server"], "MongoDB": ["mongodb", "mongo db"],
    "Redis": ["redis"], "Cassandra": ["cassandra"], "NoSQL": ["nosql"], "SQLite": ["sqlite"],
    # data / ML
    "Machine Learning": ["machine learning", "ml"], "Deep Learning": ["deep learning"], "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision", "opencv"], "TensorFlow": ["tensorflow"], "PyTorch": ["pytorch"], "Keras": ["keras"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"], "Pandas": ["pandas"], "NumPy": ["numpy"],
    "Statistics": ["statistics", "statistical analysis", "statistical modeling", "statistical modelling"],
    "Data Analysis": ["data analysis", "data analytics", "analytics"], "Data Mining": ["data mining"], "Data Visualization": ["data visualization", "data visualisation"],
    "Tableau": ["tableau"], "Power BI": ["power bi", "powerbi"], "Excel": ["excel", "ms excel", "advanced excel", "microsoft excel"],
    "Hadoop": ["hadoop", "hdfs", "mapreduce"], "Spark": ["spark", "pyspark", "apache spark"], "Hive": ["hive"], "Kafka": ["kafka"],
    "ETL": ["etl"], "Data Warehousing": ["data warehousing", "data warehouse", "dwh"], "Big Data": ["big data"], "SAS": ["sas"],
    "Artificial Intelligence": ["artificial intelligence"], "Data Science": ["data science"], "A/B Testing": ["a/b testing", "ab testing"],
    "Business Intelligence": ["business intelligence", "bi tools"], "SPSS": ["spss"],
    # cloud / devops
    "AWS": ["aws", "amazon web services", "ec2", "s3"], "Azure": ["azure", "microsoft azure"], "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"], "Kubernetes": ["kubernetes", "k8s"], "Jenkins": ["jenkins"], "Git": ["git", "github", "gitlab"],
    "CI/CD": ["ci/cd", "continuous integration", "cicd"], "Linux": ["linux", "unix"], "DevOps": ["devops"], "Terraform": ["terraform"],
    "Ansible": ["ansible"], "Cloud Computing": ["cloud computing"], "Networking": ["networking", "tcp/ip", "lan wan"],
    "Windows Server": ["windows server"], "VMware": ["vmware"], "SAP": ["sap"], "Salesforce": ["salesforce"],
    # security
    "Cyber Security": ["cyber security", "cybersecurity", "information security", "infosec", "network security", "application security"],
    "Penetration Testing": ["penetration testing", "pentesting", "pen testing", "ethical hacking", "vapt"],
    "SIEM": ["siem", "splunk", "qradar"], "Firewall": ["firewall", "firewalls", "palo alto", "fortinet", "checkpoint"],
    "Cryptography": ["cryptography", "encryption"], "Vulnerability Assessment": ["vulnerability assessment", "vulnerability management"],
    "ISO 27001": ["iso 27001", "iso27001"], "SOC": ["soc analyst", "security operations"], "IDS/IPS": ["ids/ips", "intrusion detection"],
    "Wireshark": ["wireshark"], "Nmap": ["nmap"], "Metasploit": ["metasploit"], "Burp Suite": ["burp suite", "burpsuite"],
    # testing / process
    "Software Testing": ["software testing", "manual testing", "qa", "quality assurance", "test cases"], "Selenium": ["selenium"],
    "Automation Testing": ["automation testing", "test automation"], "JIRA": ["jira"], "Agile": ["agile", "scrum"],
    "Project Management": ["project management"], "SDLC": ["sdlc"], "OOP": ["oop", "oops", "object oriented", "object-oriented"],
    "Data Structures": ["data structures", "algorithms"], "Microservices": ["microservices"], "Hibernate": ["hibernate"],
    "Struts": ["struts"], "JSP": ["jsp", "servlets"], "Unit Testing": ["unit testing", "junit"], "Embedded C": ["embedded systems", "embedded c"],
    # business / general
    "Communication": ["communication skills", "communication", "verbal communication"], "Leadership": ["leadership"],
    "Problem Solving": ["problem solving", "problem-solving", "analytical skills"], "Teamwork": ["teamwork", "team player"],
    "Sales": ["sales", "business development"], "Marketing": ["marketing", "digital marketing"], "SEO": ["seo", "search engine optimization"],
    "Content Writing": ["content writing", "copywriting"], "Accounting": ["accounting", "tally", "accounts payable", "accounts receivable"],
    "Customer Service": ["customer service", "customer support", "customer care"], "MS Office": ["ms office", "microsoft office", "ms word", "powerpoint"],
    "Recruitment": ["recruitment", "talent acquisition"], "Operations": ["operations management"], "Finance": ["financial analysis", "financial reporting"],
    "Graphic Design": ["graphic design", "corel draw", "coreldraw", "illustrator"], "UI/UX": ["ui/ux", "ux design", "ui design", "user interface"],
    "Embedded Systems": [], "Networking Security": [],
}
# Very short / ambiguous names that need case-sensitive or extra-strict matching
_STRICT = {"C": r"(?<![A-Za-z0-9+#.])C(?![A-Za-z0-9+#]|\s*(?:#|\+\+))(?:\s*(?:language|programming|,|/|$))",
           "R": r"(?<![A-Za-z0-9+#.])R(?:\s+(?:programming|language)|\s*,\s*(?:Python|SAS|SQL)|\s+and\s+Python)"}
_DROP = {"Embedded Systems", "Networking Security"}


def normalize_skill(name: str) -> str:
    """Map any alias to its canonical skill name (returns input title-cased-as-is if unknown)."""
    key = name.lower().strip()
    for canon, aliases in SKILL_ALIASES.items():
        if key == canon.lower() or key in aliases:
            return canon
    return name.strip()


def _build_pattern():
    alias_to_canon = {}
    for canon, aliases in SKILL_ALIASES.items():
        if canon in _DROP or canon in _STRICT:
            continue
        for a in set(aliases + [canon.lower()]):
            alias_to_canon[a] = canon
    parts = sorted(alias_to_canon, key=len, reverse=True)
    pat = "|".join(re.escape(p) for p in parts)
    rx = re.compile(r"(?<![a-z0-9+#.])(" + pat + r")(?![a-z0-9+#]|\.[a-z0-9])", re.I)
    return rx, alias_to_canon


_RX, _ALIAS = _build_pattern()
_STRICT_RX = {k: re.compile(v) for k, v in _STRICT.items()}
ALL_SKILLS = sorted({c for c in SKILL_ALIASES if c not in _DROP})


def extract_skills(text) -> list:
    """Return sorted list of canonical skills found in text (empty list if none / bad input)."""
    if not isinstance(text, str) or not text.strip():
        return []
    t = text.replace("\xa0", " ")
    found = {_ALIAS[m.group(1).lower()] for m in _RX.finditer(t)}
    for canon, rx in _STRICT_RX.items():
        if rx.search(t):
            found.add(canon)
    return sorted(found)


# General business / soft skills (kept for matching, but excluded from the "technical skills" dashboard view)
NON_TECH = {"Communication", "Leadership", "Problem Solving", "Teamwork", "Sales", "Marketing", "Content Writing", "Accounting",
            "Customer Service", "MS Office", "Recruitment", "Operations", "Finance", "Graphic Design", "Project Management", "SEO", "Agile"}
