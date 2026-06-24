document.addEventListener('DOMContentLoaded', () => {
    // 仓库根前缀，你的站点是 /zipformer/
    const basePrefix = '/zipformer/';
    // 语言列表映射
    const langs = ['en', ''];

    // 拦截所有语言切换下拉点击
    const langLinks = document.querySelectorAll('.md-header__option[href]');
    langLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetLangUrl = this.getAttribute('href');
            // 提取目标语言 en / zh
            const targetLang = targetLangUrl.replace(basePrefix, '').replace('/', '');

            // 获取当前完整路径，去除基础前缀
            let currentPath = location.pathname.replace(basePrefix, '');
            // 移除当前语言前缀
            langs.forEach(l => {
                currentPath = currentPath.replace(`${l}/`, '');
            });

            // 拼接新地址：/zipformer/{lang}/{剩余路径}
            const newUrl = `${basePrefix}${targetLang}/${currentPath}`;
            location.href = newUrl;
        });
    });
});